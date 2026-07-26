"""
vcf_utils.py
Module for extracting and formatting VCF data for LLM-powered analysis.
Uses cyvcf2 for efficient VCF parsing.
"""

from cyvcf2 import VCF
from typing import Dict, Any, List, Optional
import os

# Kuzu integration imports
from . import graph_integration # Assuming graph_integration.py is in the same directory
import kuzu # For kuzu.Connection type hint if needed directly


def extract_variant_summary(vcf_path: str) -> Dict[str, Any]:
    """
    Extracts summary statistics from a VCF file.
    Returns a dictionary with variant counts, types, and sample stats.
    """
    if not os.path.isfile(vcf_path):
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    vcf = VCF(vcf_path)
    variant_count = 0
    variant_types = {}
    samples = vcf.samples
    sample_counts = {sample: 0 for sample in samples}

    for record in vcf:
        variant_count += 1
        vtype = record.var_type  # e.g., 'snp', 'indel', etc.
        variant_types[vtype] = variant_types.get(vtype, 0) + 1
        # Count variants per sample (GT field)
        for i, gt in enumerate(record.genotypes):
            # gt is a list like [0, 1, True] (last is phased)
            if any(allele > 0 for allele in gt[:2]):
                sample_counts[samples[i]] += 1

    return {
        "variant_count": variant_count,
        "variant_types": variant_types,
        "samples": samples,
        "sample_variant_counts": sample_counts,
    }


def extract_comparable_features(vcf_path: str) -> Dict[str, Any]:
    """
    Extracts features from a VCF file for comparison tasks.
    Returns a dictionary with key features for side-by-side comparison.
    """
    summary = extract_variant_summary(vcf_path)
    # Add more features as needed for comparison
    return summary


def get_sample_names(vcf_path: str) -> List[str]:
    """
    Returns the list of sample names in the VCF file.
    """
    vcf = VCF(vcf_path)
    return vcf.samples


def populate_kuzu_from_vcf(kuzu_conn: kuzu.Connection, vcf_path: str, sample_name_override: Optional[str] = None) -> Dict[str, int]:
    """
    Parses a VCF file and populates the Kuzu graph database with variants, samples, and their relationships.

    Args:
        kuzu_conn: Active Kuzu connection object.
        vcf_path: Path to the VCF file.
        sample_name_override: Optional. If provided, use this as the sample_id for all records in the VCF,
                              useful for single-sample VCFs or when a specific ID is desired.
                              If None, uses sample names from the VCF header.

    Returns:
        A dictionary with counts of added variants, samples, and links.
    """
    if not os.path.isfile(vcf_path):
        raise FileNotFoundError(f"VCF file not found: {vcf_path}")

    vcf_parser = VCF(vcf_path)
    
    counts = {"variants_added": 0, "samples_added": 0, "links_added": 0}
    processed_samples = set() # To track samples already added in this run

    vcf_samples = vcf_parser.samples
    if sample_name_override:
        # If overriding, ensure the single sample is added to Kuzu
        if sample_name_override not in processed_samples:
            try:
                graph_integration.add_sample(kuzu_conn, {'sample_id': sample_name_override})
                counts["samples_added"] += 1
                processed_samples.add(sample_name_override)
            except Exception as e: # Catch potential Kuzu errors (e.g., if already exists, though add_sample should be idempotent)
                print(f"Info: Could not add sample {sample_name_override} (may already exist or error): {e}")
    else:
        # Add all samples from VCF header if no override
        for s_name in vcf_samples:
            if s_name not in processed_samples:
                try:
                    graph_integration.add_sample(kuzu_conn, {'sample_id': s_name})
                    counts["samples_added"] += 1
                    processed_samples.add(s_name)
                except Exception as e:
                    print(f"Info: Could not add sample {s_name} (may already exist or error): {e}")

    for record in vcf_parser:
        variant_id = f"{record.CHROM}-{record.POS}-{record.REF}-{record.ALT[0]}" # Assuming first ALT allele for simplicity
        rs_id = record.ID if record.ID else None

        variant_data = {
            'variant_id': variant_id,
            'chrom': record.CHROM,
            'pos': record.POS,
            'ref': record.REF,
            'alt': record.ALT[0], # Simplification: taking the first alternate allele
            'rs_id': rs_id
        }
        try:
            graph_integration.add_variant(kuzu_conn, variant_data)
            counts["variants_added"] += 1
        except Exception as e: # Catch potential Kuzu errors
            print(f"Info: Could not add variant {variant_id} (may already exist or error): {e}")
            # Continue to attempt linking if variant might exist

        # Genotypes: record.genotypes is a list of lists, e.g. [[0, 1, True], [1, 1, False]]
        # Each inner list: [allele1_idx, allele2_idx, phased_boolean]
        # Allele idx: 0=REF, 1=ALT1, 2=ALT2 etc.
        for i, gt_info in enumerate(record.genotypes):
            # Determine sample ID for this genotype
            current_sample_id = sample_name_override if sample_name_override else vcf_samples[i]

            allele1 = gt_info[0]
            allele2 = gt_info[1]
            
            zygosity = "UNKNOWN"
            if allele1 == 0 and allele2 == 0: # REF/REF
                zygosity = "HOM_REF" # Not typically stored as an "ObservedIn" link for the ALT
                continue # Skip HOM_REF as we are tracking observations of alternate alleles
            elif allele1 == 0 and allele2 > 0: # REF/ALT
                zygosity = "HET"
            elif allele1 > 0 and allele2 == 0: # ALT/REF (also HET)
                zygosity = "HET"
            elif allele1 > 0 and allele2 > 0:
                if allele1 == allele2: # ALT/ALT (same ALT)
                    zygosity = "HOM_ALT"
                else: # ALT1/ALT2 (different ALTs, complex case)
                    zygosity = "HET_ALT_MULTIPLE" # Or some other designation for multi-allelic hets
            
            # We only link if at least one ALT allele is present for this variant
            # And we are simplifying to the first ALT allele for this variant_id
            if any(a > 0 for a in [allele1, allele2]): # Checks if any allele index is > 0 (i.e., an ALT)
                try:
                    link_props = {'zygosity': zygosity}
                    graph_integration.link_variant_to_sample(kuzu_conn, current_sample_id, variant_id, link_props)
                    counts["links_added"] += 1
                except Exception as e:
                    print(f"Info: Could not link variant {variant_id} to sample {current_sample_id}: {e}")
    
    vcf_parser.close()
    # Return with keys matching test expectations
    return {
        "variants": counts["variants_added"],
        "samples": counts["samples_added"],
        "links": counts["links_added"]
    } 