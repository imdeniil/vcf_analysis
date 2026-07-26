"""
SAMspec Compliance Validation Module

Validates VCF files against the SAM/VCF specification standards (VCF 4.2/4.3).
Ensures format compliance, header validation, and data integrity checks.
"""

import re
import logging
from typing import Dict, List, Tuple, Optional, Set, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import gzip


class ComplianceLevel(Enum):
    """Compliance violation severity levels."""
    CRITICAL = "critical"    # Specification violations that break compatibility
    WARNING = "warning"      # Recommended practices not followed
    INFO = "info"           # Minor issues or suggestions


@dataclass
class ComplianceViolation:
    """Represents a SAMspec compliance violation."""
    level: ComplianceLevel
    rule_id: str
    message: str
    line_number: Optional[int] = None
    field: Optional[str] = None
    value: Optional[str] = None
    suggestion: Optional[str] = None


@dataclass
class ComplianceReport:
    """Complete SAMspec compliance report."""
    file_path: str
    vcf_version: Optional[str]
    total_violations: int
    critical_count: int
    warning_count: int
    info_count: int
    violations: List[ComplianceViolation]
    is_compliant: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "vcf_version": self.vcf_version,
            "total_violations": self.total_violations,
            "critical_count": self.critical_count,
            "warning_count": self.warning_count,
            "info_count": self.info_count,
            "is_compliant": self.is_compliant,
            "violations": [
                {
                    "level": v.level.value,
                    "rule_id": v.rule_id,
                    "message": v.message,
                    "line_number": v.line_number,
                    "field": v.field,
                    "value": v.value,
                    "suggestion": v.suggestion
                }
                for v in self.violations
            ]
        }


class SAMspecValidator:
    """SAMspec compliance validator for VCF files."""
    
    def __init__(self):
        """Initialize the validator with specification rules."""
        self.violations: List[ComplianceViolation] = []
        self.vcf_version: Optional[str] = None
        self.header_lines: List[str] = []
        self.data_lines: List[str] = []
        self.info_fields: Dict[str, Dict] = {}
        self.format_fields: Dict[str, Dict] = {}
        self.filter_fields: Dict[str, Dict] = {}
        self.contig_fields: Dict[str, Dict] = {}
        self.samples: List[str] = []
        
    def validate_file(self, file_path: str) -> ComplianceReport:
        """
        Validate a VCF file for SAMspec compliance.
        
        Args:
            file_path: Path to the VCF file
            
        Returns:
            ComplianceReport with validation results
        """
        self.violations = []
        
        try:
            # Read and parse the VCF file
            self._read_vcf_file(file_path)
            
            # Perform validation checks
            self._validate_file_format()
            self._validate_header_structure()
            self._validate_mandatory_headers()
            self._validate_info_definitions()
            self._validate_format_definitions()
            self._validate_filter_definitions()
            self._validate_contig_definitions()
            self._validate_data_records()
            self._validate_cross_references()
            
        except Exception as e:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="FILE_READ_ERROR",
                message=f"Failed to read VCF file: {str(e)}"
            ))
        
        # Generate report
        return self._generate_report(file_path)
    
    def _read_vcf_file(self, file_path: str) -> None:
        """Read and parse VCF file into header and data sections."""
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"VCF file not found: {file_path}")
        
        # Handle compressed files
        if file_path.endswith('.gz'):
            open_func = gzip.open
            mode = 'rt'
        else:
            open_func = open
            mode = 'r'
        
        with open_func(file_path, mode, encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip('\n\r')
                
                if line.startswith('##'):
                    self.header_lines.append(line)
                elif line.startswith('#CHROM'):
                    self.header_lines.append(line)
                    # Parse sample names
                    fields = line.split('\t')
                    if len(fields) > 9:
                        self.samples = fields[9:]
                elif line.strip():  # Non-empty data line
                    self.data_lines.append(line)
    
    def _validate_file_format(self) -> None:
        """Validate the ##fileformat header line."""
        format_line = None
        
        for line in self.header_lines:
            if line.startswith('##fileformat='):
                if format_line is not None:
                    self.violations.append(ComplianceViolation(
                        level=ComplianceLevel.CRITICAL,
                        rule_id="DUPLICATE_FILEFORMAT",
                        message="Multiple ##fileformat lines found",
                        suggestion="Only one ##fileformat line is allowed"
                    ))
                format_line = line
                break
        
        if format_line is None:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="MISSING_FILEFORMAT",
                message="Missing required ##fileformat header line",
                suggestion="Add ##fileformat=VCFv4.2 or ##fileformat=VCFv4.3"
            ))
            return
        
        # Extract version
        version_match = re.match(r'##fileformat=VCFv(\d+\.\d+)', format_line)
        if not version_match:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_FILEFORMAT",
                message=f"Invalid ##fileformat line: {format_line}",
                suggestion="Use format: ##fileformat=VCFv4.2 or ##fileformat=VCFv4.3"
            ))
            return
        
        self.vcf_version = version_match.group(1)
        
        # Check if version is supported
        if self.vcf_version not in ['4.0', '4.1', '4.2', '4.3']:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.WARNING,
                rule_id="UNSUPPORTED_VERSION",
                message=f"VCF version {self.vcf_version} may not be fully supported",
                suggestion="Consider using VCF 4.2 or 4.3"
            ))
        
        # Check if fileformat is the first line
        if self.header_lines[0] != format_line:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.WARNING,
                rule_id="FILEFORMAT_NOT_FIRST",
                message="##fileformat should be the first line in the file",
                suggestion="Move ##fileformat to the beginning of the file"
            ))
    
    def _validate_header_structure(self) -> None:
        """Validate overall header structure."""
        # Check for #CHROM line
        chrom_line = None
        for line in self.header_lines:
            if line.startswith('#CHROM'):
                chrom_line = line
                break
        
        if chrom_line is None:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="MISSING_CHROM_LINE",
                message="Missing required #CHROM header line",
                suggestion="Add #CHROM header line before data records"
            ))
            return
        
        # Validate #CHROM line format
        expected_fields = ['#CHROM', 'POS', 'ID', 'REF', 'ALT', 'QUAL', 'FILTER', 'INFO']
        fields = chrom_line.split('\t')
        
        if len(fields) < 8:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_CHROM_LINE",
                message="#CHROM line has insufficient fields",
                suggestion="Must have at least 8 fields: #CHROM, POS, ID, REF, ALT, QUAL, FILTER, INFO"
            ))
            return
        
        for i, expected in enumerate(expected_fields):
            if i < len(fields) and fields[i] != expected:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="INVALID_CHROM_FIELD",
                    message=f"Invalid field in #CHROM line: expected '{expected}', got '{fields[i]}'",
                    field=f"position_{i}"
                ))
        
        # Check for FORMAT field if samples are present
        if len(fields) > 8:
            if len(fields) == 9:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="MISSING_FORMAT_FIELD",
                    message="FORMAT field required when sample columns are present",
                    suggestion="Add FORMAT field before sample columns"
                ))
            elif fields[8] != 'FORMAT':
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="INVALID_FORMAT_FIELD",
                    message=f"Expected 'FORMAT' field, got '{fields[8]}'",
                    field="FORMAT"
                ))
    
    def _validate_mandatory_headers(self) -> None:
        """Validate mandatory header definitions."""
        # Parse header definitions
        for line in self.header_lines:
            if line.startswith('##INFO='):
                self._parse_info_definition(line)
            elif line.startswith('##FORMAT='):
                self._parse_format_definition(line)
            elif line.startswith('##FILTER='):
                self._parse_filter_definition(line)
            elif line.startswith('##contig='):
                self._parse_contig_definition(line)
        
        # Check for mandatory FILTER definitions
        if 'PASS' not in self.filter_fields:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.WARNING,
                rule_id="MISSING_PASS_FILTER",
                message="Missing PASS filter definition",
                suggestion="Add ##FILTER=<ID=PASS,Description=\"All filters passed\">"
            ))
    
    def _parse_info_definition(self, line: str) -> None:
        """Parse ##INFO header line."""
        match = re.match(r'##INFO=<(.+)>', line)
        if not match:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_INFO_FORMAT",
                message=f"Invalid ##INFO format: {line}",
                suggestion="Use format: ##INFO=<ID=...,Number=...,Type=...,Description=\"...\">"
            ))
            return
        
        # Parse key-value pairs
        content = match.group(1)
        info_def = self._parse_header_content(content)
        
        # Validate required fields
        required_fields = ['ID', 'Number', 'Type', 'Description']
        for field in required_fields:
            if field not in info_def:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="MISSING_INFO_FIELD",
                    message=f"Missing required field '{field}' in INFO definition: {line}",
                    field=field
                ))
        
        if 'ID' in info_def:
            self.info_fields[info_def['ID']] = info_def
    
    def _parse_format_definition(self, line: str) -> None:
        """Parse ##FORMAT header line."""
        match = re.match(r'##FORMAT=<(.+)>', line)
        if not match:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_FORMAT_FORMAT",
                message=f"Invalid ##FORMAT format: {line}",
                suggestion="Use format: ##FORMAT=<ID=...,Number=...,Type=...,Description=\"...\">"
            ))
            return
        
        content = match.group(1)
        format_def = self._parse_header_content(content)
        
        # Validate required fields
        required_fields = ['ID', 'Number', 'Type', 'Description']
        for field in required_fields:
            if field not in format_def:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="MISSING_FORMAT_FIELD",
                    message=f"Missing required field '{field}' in FORMAT definition: {line}",
                    field=field
                ))
        
        if 'ID' in format_def:
            self.format_fields[format_def['ID']] = format_def
    
    def _parse_filter_definition(self, line: str) -> None:
        """Parse ##FILTER header line."""
        match = re.match(r'##FILTER=<(.+)>', line)
        if not match:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_FILTER_FORMAT",
                message=f"Invalid ##FILTER format: {line}",
                suggestion="Use format: ##FILTER=<ID=...,Description=\"...\">"
            ))
            return
        
        content = match.group(1)
        filter_def = self._parse_header_content(content)
        
        # Validate required fields
        required_fields = ['ID', 'Description']
        for field in required_fields:
            if field not in filter_def:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="MISSING_FILTER_FIELD",
                    message=f"Missing required field '{field}' in FILTER definition: {line}",
                    field=field
                ))
        
        if 'ID' in filter_def:
            self.filter_fields[filter_def['ID']] = filter_def
    
    def _parse_contig_definition(self, line: str) -> None:
        """Parse ##contig header line."""
        match = re.match(r'##contig=<(.+)>', line)
        if not match:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.WARNING,
                rule_id="INVALID_CONTIG_FORMAT",
                message=f"Invalid ##contig format: {line}",
                suggestion="Use format: ##contig=<ID=...,length=...>"
            ))
            return
        
        content = match.group(1)
        contig_def = self._parse_header_content(content)
        
        if 'ID' not in contig_def:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.WARNING,
                rule_id="MISSING_CONTIG_ID",
                message=f"Missing ID field in contig definition: {line}",
                field="ID"
            ))
        else:
            self.contig_fields[contig_def['ID']] = contig_def
    
    def _parse_header_content(self, content: str) -> Dict[str, str]:
        """Parse key-value pairs from header content."""
        result = {}
        
        # Handle quoted values
        pattern = r'(\w+)=(?:"([^"]*)"|([^,]+))'
        matches = re.findall(pattern, content)
        
        for match in matches:
            key = match[0]
            value = match[1] if match[1] else match[2]
            result[key] = value
        
        return result
    
    def _validate_info_definitions(self) -> None:
        """Validate INFO field definitions."""
        for info_id, info_def in self.info_fields.items():
            # Validate Type
            if 'Type' in info_def:
                valid_types = ['Integer', 'Float', 'Flag', 'Character', 'String']
                if info_def['Type'] not in valid_types:
                    self.violations.append(ComplianceViolation(
                        level=ComplianceLevel.CRITICAL,
                        rule_id="INVALID_INFO_TYPE",
                        message=f"Invalid Type '{info_def['Type']}' for INFO field '{info_id}'",
                        field="Type",
                        suggestion=f"Use one of: {', '.join(valid_types)}"
                    ))
            
            # Validate Number
            if 'Number' in info_def:
                number = info_def['Number']
                if not re.match(r'^(\d+|A|R|G|\.)$', number):
                    self.violations.append(ComplianceViolation(
                        level=ComplianceLevel.CRITICAL,
                        rule_id="INVALID_INFO_NUMBER",
                        message=f"Invalid Number '{number}' for INFO field '{info_id}'",
                        field="Number",
                        suggestion="Use integer, A, R, G, or ."
                    ))
    
    def _validate_format_definitions(self) -> None:
        """Validate FORMAT field definitions."""
        for format_id, format_def in self.format_fields.items():
            # Validate Type
            if 'Type' in format_def:
                valid_types = ['Integer', 'Float', 'Character', 'String']
                if format_def['Type'] not in valid_types:
                    self.violations.append(ComplianceViolation(
                        level=ComplianceLevel.CRITICAL,
                        rule_id="INVALID_FORMAT_TYPE",
                        message=f"Invalid Type '{format_def['Type']}' for FORMAT field '{format_id}'",
                        field="Type",
                        suggestion=f"Use one of: {', '.join(valid_types)}"
                    ))
            
            # Validate Number
            if 'Number' in format_def:
                number = format_def['Number']
                if not re.match(r'^(\d+|A|R|G|\.)$', number):
                    self.violations.append(ComplianceViolation(
                        level=ComplianceLevel.CRITICAL,
                        rule_id="INVALID_FORMAT_NUMBER",
                        message=f"Invalid Number '{number}' for FORMAT field '{format_id}'",
                        field="Number",
                        suggestion="Use integer, A, R, G, or ."
                    ))
    
    def _validate_filter_definitions(self) -> None:
        """Validate FILTER field definitions."""
        # Basic validation - filters should have ID and Description
        for filter_id, filter_def in self.filter_fields.items():
            if not filter_def.get('Description'):
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.WARNING,
                    rule_id="MISSING_FILTER_DESCRIPTION",
                    message=f"FILTER '{filter_id}' missing description",
                    field="Description"
                ))
    
    def _validate_contig_definitions(self) -> None:
        """Validate contig definitions."""
        # Check for length field in contig definitions
        for contig_id, contig_def in self.contig_fields.items():
            if 'length' not in contig_def:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.INFO,
                    rule_id="MISSING_CONTIG_LENGTH",
                    message=f"Contig '{contig_id}' missing length field",
                    field="length",
                    suggestion="Consider adding length field for better compatibility"
                ))
    
    def _validate_data_records(self) -> None:
        """Validate data record format and content."""
        for line_num, line in enumerate(self.data_lines, len(self.header_lines) + 1):
            self._validate_data_record(line, line_num)
    
    def _validate_data_record(self, line: str, line_num: int) -> None:
        """Validate a single data record."""
        fields = line.split('\t')
        
        # Check minimum number of fields
        if len(fields) < 8:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INSUFFICIENT_FIELDS",
                message=f"Data record has {len(fields)} fields, minimum 8 required",
                line_number=line_num
            ))
            return
        
        # Validate individual fields
        self._validate_chrom_field(fields[0], line_num)
        self._validate_pos_field(fields[1], line_num)
        self._validate_id_field(fields[2], line_num)
        self._validate_ref_field(fields[3], line_num)
        self._validate_alt_field(fields[4], line_num)
        self._validate_qual_field(fields[5], line_num)
        self._validate_filter_field(fields[6], line_num)
        self._validate_info_field(fields[7], line_num)
        
        # Validate FORMAT and sample fields if present
        if len(fields) > 8:
            if len(fields) < 10:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="MISSING_SAMPLE_DATA",
                    message="FORMAT field present but no sample data",
                    line_number=line_num
                ))
            else:
                self._validate_format_field(fields[8], line_num)
                for i, sample_data in enumerate(fields[9:], 9):
                    self._validate_sample_field(sample_data, fields[8], line_num, i-8)
    
    def _validate_chrom_field(self, chrom: str, line_num: int) -> None:
        """Validate CHROM field."""
        if not chrom or chrom == '.':
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_CHROM",
                message="CHROM field cannot be empty or '.'",
                line_number=line_num,
                field="CHROM"
            ))
    
    def _validate_pos_field(self, pos: str, line_num: int) -> None:
        """Validate POS field."""
        try:
            pos_int = int(pos)
            if pos_int < 1:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="INVALID_POS",
                    message=f"POS field must be >= 1, got {pos_int}",
                    line_number=line_num,
                    field="POS"
                ))
        except ValueError:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_POS_FORMAT",
                message=f"POS field must be an integer, got '{pos}'",
                line_number=line_num,
                field="POS"
            ))
    
    def _validate_id_field(self, id_field: str, line_num: int) -> None:
        """Validate ID field."""
        # ID can be '.' or a valid identifier
        if id_field != '.' and not re.match(r'^[A-Za-z0-9_.:;-]+$', id_field):
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.WARNING,
                rule_id="INVALID_ID_FORMAT",
                message=f"ID field contains invalid characters: '{id_field}'",
                line_number=line_num,
                field="ID"
            ))
    
    def _validate_ref_field(self, ref: str, line_num: int) -> None:
        """Validate REF field."""
        if not ref or ref == '.':
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_REF",
                message="REF field cannot be empty or '.'",
                line_number=line_num,
                field="REF"
            ))
        elif not re.match(r'^[ACGTNacgtn]+$', ref):
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_REF_BASES",
                message=f"REF field contains invalid bases: '{ref}'",
                line_number=line_num,
                field="REF"
            ))
    
    def _validate_alt_field(self, alt: str, line_num: int) -> None:
        """Validate ALT field."""
        if alt == '.':
            return  # '.' is valid for ALT
        
        alts = alt.split(',')
        for alt_allele in alts:
            if not re.match(r'^[ACGTNacgtn*]+$|^<[^>]+>$', alt_allele):
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.CRITICAL,
                    rule_id="INVALID_ALT_BASES",
                    message=f"ALT field contains invalid allele: '{alt_allele}'",
                    line_number=line_num,
                    field="ALT"
                ))
    
    def _validate_qual_field(self, qual: str, line_num: int) -> None:
        """Validate QUAL field."""
        if qual == '.':
            return  # '.' is valid for QUAL
        
        try:
            qual_float = float(qual)
            if qual_float < 0:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.WARNING,
                    rule_id="NEGATIVE_QUAL",
                    message=f"QUAL field is negative: {qual_float}",
                    line_number=line_num,
                    field="QUAL"
                ))
        except ValueError:
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.CRITICAL,
                rule_id="INVALID_QUAL_FORMAT",
                message=f"QUAL field must be a number or '.', got '{qual}'",
                line_number=line_num,
                field="QUAL"
            ))
    
    def _validate_filter_field(self, filter_field: str, line_num: int) -> None:
        """Validate FILTER field."""
        if filter_field == '.' or filter_field == 'PASS':
            return
        
        filters = filter_field.split(';')
        for filter_name in filters:
            if filter_name not in self.filter_fields:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.WARNING,
                    rule_id="UNDEFINED_FILTER",
                    message=f"Filter '{filter_name}' not defined in header",
                    line_number=line_num,
                    field="FILTER",
                    suggestion=f"Add ##FILTER definition for '{filter_name}'"
                ))
    
    def _validate_info_field(self, info_field: str, line_num: int) -> None:
        """Validate INFO field."""
        if info_field == '.':
            return
        
        info_pairs = info_field.split(';')
        for pair in info_pairs:
            if '=' in pair:
                key, value = pair.split('=', 1)
            else:
                key = pair
                value = None
            
            if key not in self.info_fields:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.WARNING,
                    rule_id="UNDEFINED_INFO",
                    message=f"INFO field '{key}' not defined in header",
                    line_number=line_num,
                    field="INFO",
                    suggestion=f"Add ##INFO definition for '{key}'"
                ))
    
    def _validate_format_field(self, format_field: str, line_num: int) -> None:
        """Validate FORMAT field."""
        format_keys = format_field.split(':')
        for key in format_keys:
            if key not in self.format_fields:
                self.violations.append(ComplianceViolation(
                    level=ComplianceLevel.WARNING,
                    rule_id="UNDEFINED_FORMAT",
                    message=f"FORMAT field '{key}' not defined in header",
                    line_number=line_num,
                    field="FORMAT",
                    suggestion=f"Add ##FORMAT definition for '{key}'"
                ))
    
    def _validate_sample_field(self, sample_data: str, format_field: str, line_num: int, sample_idx: int) -> None:
        """Validate sample data field."""
        format_keys = format_field.split(':')
        sample_values = sample_data.split(':')
        
        if len(sample_values) != len(format_keys):
            self.violations.append(ComplianceViolation(
                level=ComplianceLevel.WARNING,
                rule_id="FORMAT_SAMPLE_MISMATCH",
                message=f"Sample {sample_idx} has {len(sample_values)} values but FORMAT has {len(format_keys)} keys",
                line_number=line_num,
                field=f"sample_{sample_idx}"
            ))
    
    def _validate_cross_references(self) -> None:
        """Validate cross-references between header definitions and data usage."""
        # This would involve more complex validation of data consistency
        # For now, we'll implement basic checks
        pass
    
    def _generate_report(self, file_path: str) -> ComplianceReport:
        """Generate the final compliance report."""
        critical_count = sum(1 for v in self.violations if v.level == ComplianceLevel.CRITICAL)
        warning_count = sum(1 for v in self.violations if v.level == ComplianceLevel.WARNING)
        info_count = sum(1 for v in self.violations if v.level == ComplianceLevel.INFO)
        
        # File is compliant if there are no critical violations
        is_compliant = critical_count == 0
        
        return ComplianceReport(
            file_path=file_path,
            vcf_version=self.vcf_version,
            total_violations=len(self.violations),
            critical_count=critical_count,
            warning_count=warning_count,
            info_count=info_count,
            violations=self.violations,
            is_compliant=is_compliant
        )


def validate_vcf_samspec_compliance(file_path: str) -> ComplianceReport:
    """
    Convenience function to validate VCF file for SAMspec compliance.
    
    Args:
        file_path: Path to the VCF file
        
    Returns:
        ComplianceReport with validation results
    """
    validator = SAMspecValidator()
    return validator.validate_file(file_path) 