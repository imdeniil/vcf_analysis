"""
Enhanced Kuzu Graph Database Integration for VCF Agent

This module provides comprehensive functions to interact with a Kuzu graph database,
implementing the DECISION-001 specifications for genomic relationship modeling.
Includes schema definition, data loading, and optimized querying for VCF analysis.
"""
import os
import kuzu
import pandas as pd # Added for DataFrame conversion
import pyarrow as pa # Ensure pyarrow is imported
from typing import Dict, List, Any, Optional, Union, Tuple
import logging
from datetime import datetime
import time
from concurrent.futures import ThreadPoolExecutor
import threading

# Configure logging
logger = logging.getLogger(__name__)

# Default path for the Kuzu database. Honours the KUZU_PATH env var set in the
# docker-compose runtime (typically a mounted volume directory, e.g. /app/kuzu_db).
DEFAULT_KUZU_DB_PATH = os.getenv("KUZU_PATH", "./kuzu_db")

# Global variable to hold the managed Kuzu connection
_kuzu_main_connection: Optional[kuzu.Connection] = None
_kuzu_lock = threading.Lock()  # Thread safety for connection management

# Enhanced schema based on DECISION-001 specifications
ENHANCED_SCHEMA_QUERIES = [
    # Sample node table
    """CREATE NODE TABLE IF NOT EXISTS Sample(
        id STRING, 
        name STRING, 
        type STRING, 
        created_at TIMESTAMP,
        metadata STRING,
        PRIMARY KEY (id)
    )""",
    
    # Variant node table with comprehensive genomic information
    """CREATE NODE TABLE IF NOT EXISTS Variant(
        id STRING, 
        chr STRING, 
        pos INT64, 
        ref STRING, 
        alt STRING,
        variant_type STRING,
        quality_score DOUBLE,
        filter_status STRING,
        allele_frequency DOUBLE,
        created_at TIMESTAMP,
        PRIMARY KEY (id)
    )""",
    
    # Gene node table for genomic context
    """CREATE NODE TABLE IF NOT EXISTS Gene(
        id STRING, 
        symbol STRING, 
        name STRING, 
        chromosome STRING,
        start_pos INT64,
        end_pos INT64,
        biotype STRING,
        PRIMARY KEY (id)
    )""",
    
    # Analysis node table for AI-generated insights
    """CREATE NODE TABLE IF NOT EXISTS Analysis(
        id STRING,
        type STRING,
        summary STRING,
        confidence_score DOUBLE,
        created_at TIMESTAMP,
        PRIMARY KEY (id)
    )""",
    
    # HasVariant relationship: Sample -> Variant
    """CREATE REL TABLE IF NOT EXISTS HasVariant(
        FROM Sample TO Variant, 
        genotype STRING, 
        quality DOUBLE,
        depth INT64,
        allele_depth STRING,
        created_at TIMESTAMP
    )""",
    
    # LocatedIn relationship: Variant -> Gene
    """CREATE REL TABLE IF NOT EXISTS LocatedIn(
        FROM Variant TO Gene, 
        impact STRING, 
        consequence STRING,
        amino_acid_change STRING,
        codon_change STRING
    )""",
    
    # AnalyzedBy relationship: Variant -> Analysis
    """CREATE REL TABLE IF NOT EXISTS AnalyzedBy(
        FROM Variant TO Analysis,
        created_at TIMESTAMP
    )""",
    
    # SimilarTo relationship: Variant -> Variant (for similarity networks)
    """CREATE REL TABLE IF NOT EXISTS SimilarTo(
        FROM Variant TO Variant,
        similarity_score DOUBLE,
        similarity_type STRING,
        created_at TIMESTAMP
    )"""
]

def get_managed_kuzu_connection() -> kuzu.Connection:
    """
    Initializes and returns a managed Kuzu database connection.
    Ensures the connection is established once and the schema is created.

    Uses DEFAULT_KUZU_DB_PATH for the database location.

    Returns:
        A Kuzu database connection object.
    
    Raises:
        RuntimeError: If the Kuzu connection cannot be established or schema cannot be created.
    """
    global _kuzu_main_connection
    if _kuzu_main_connection is None:
        try:
            print(f"Initializing managed Kuzu connection to: {DEFAULT_KUZU_DB_PATH}")
            # Use the existing get_kuzu_db_connection which returns a new connection to a DB instance
            # This doesn't strictly create a singleton DB object across different calls to get_kuzu_db_connection
            # if db_path is different, but here we use a fixed path.
            # Kuzu's Database object itself handles the singleton nature for a given path.
            conn = get_kuzu_db_connection(db_path=DEFAULT_KUZU_DB_PATH)
            create_enhanced_schema(conn) # Use enhanced schema instead of basic schema
            _kuzu_main_connection = conn
            print("Managed Kuzu connection initialized and enhanced schema verified.")
        except Exception as e:
            # Log the error appropriately in a real application
            print(f"Failed to initialize managed Kuzu connection or create enhanced schema: {e}")
            raise RuntimeError(f"Kuzu setup failed: {e}") from e
    
    if _kuzu_main_connection is None: # Should not be reached if logic above is correct
        raise RuntimeError("Kuzu connection is unexpectedly None after initialization attempt.")
        
    return _kuzu_main_connection

def get_kuzu_db_connection(db_path: str = DEFAULT_KUZU_DB_PATH, read_only: bool = False) -> kuzu.Connection:
    """
    Initializes a Kuzu database at the given path and returns a connection.

    Args:
        db_path: Path to the Kuzu database. May be a directory (a volume mount,
            e.g. KUZU_PATH=/app/kuzu_db) or a plain file path. Kuzu >=0.11
            rejects directories, so when db_path points to an existing
            directory we transparently use `<db_path>/kuzu.database`.
        read_only: If True, opens the database in read-only mode.

    Returns:
        A Kuzu database connection object.
    """
    # Resolve a concrete database file path. Kuzu >=0.11 expects a file path,
    # not a directory; the env var KUZU_PATH typically points at a mounted
    # volume directory, so we append a filename when needed.
    # Heuristic: treat the path as a *directory* location (and append a file
    # name inside it) when it has no file extension, regardless of whether it
    # already exists. Ensure the parent directory exists so kuzu can create the
    # database file.
    resolved_path = db_path
    base = os.path.basename(db_path.rstrip("/"))
    has_extension = "." in base
    if not has_extension:
        resolved_path = os.path.join(db_path, "kuzu.database")
    parent = os.path.dirname(resolved_path)
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)
    try:
        db = kuzu.Database(resolved_path)
        # TODO: Kuzu Python API for read_only mode needs to be confirmed.
        # As of recent versions, read_only might be set at Database level or not directly via Connection.
        # For now, we assume write access by default.
        # if read_only:
        #     conn = kuzu.Connection(db, access_mode=kuzu.AccessMode.READ_ONLY) # Example, check actual API
        # else:
        conn = kuzu.Connection(db)
        print(f"Successfully connected to Kuzu database at: {resolved_path}")
        return conn
    except Exception as e:
        print(f"Error connecting to Kuzu database at {resolved_path}: {e}")
        raise

def create_schema(conn: kuzu.Connection) -> None:
    """
    Creates the necessary node and relationship tables for VCF data if they don't exist.

    Schema:
    - Variant (Node): Represents a genetic variant.
        Properties: variant_id (STRING, PK), chrom (STRING), pos (INT64),
                    ref (STRING), alt (STRING), rs_id (STRING, optional)
    - Sample (Node): Represents a sample.
        Properties: sample_id (STRING, PK)
    - ObservedIn (Relationship): Links a Variant to a Sample.
        Properties: zygosity (STRING, e.g., HOM, HET)
    """
    schema_queries = [
        "CREATE NODE TABLE IF NOT EXISTS Variant(variant_id STRING, chrom STRING, pos INT64, ref STRING, alt STRING, rs_id STRING, PRIMARY KEY (variant_id))",
        "CREATE NODE TABLE IF NOT EXISTS Sample(sample_id STRING, PRIMARY KEY (sample_id))",
        "CREATE REL TABLE IF NOT EXISTS ObservedIn(FROM Variant TO Sample, zygosity STRING)"
    ]
    try:
        for query in schema_queries:
            print(f"Executing schema query: {query}")
            qr = conn.execute(query)
            if qr: del qr # Explicitly delete QueryResult
        print("Kuzu schema created/verified successfully.")
    except Exception as e:
        print(f"Error creating Kuzu schema: {e}")
        raise

def add_variant(conn: kuzu.Connection, variant_data: Dict[str, Any]) -> None:
    """
    Adds a single variant node to the Kuzu database using parameterized queries.

    Args:
        conn: Active Kuzu connection.
        variant_data: Dictionary containing variant properties (variant_id, chrom, pos, ref, alt, rs_id).
                      Example: {'variant_id': 'chr1-123-A-G', 'chrom': '1', 'pos': 123,
                                'ref': 'A', 'alt': 'G', 'rs_id': 'rs12345'}
    """
    try:
        query = """
        CREATE (v:Variant {
            variant_id: $variant_id,
            chrom: $chrom,
            pos: $pos,
            ref: $ref,
            alt: $alt,
            rs_id: $rs_id
        })
        """
        params = {
            "variant_id": variant_data['variant_id'],
            "chrom": variant_data['chrom'],
            "pos": variant_data['pos'],
            "ref": variant_data['ref'],
            "alt": variant_data['alt'],
            "rs_id": variant_data.get('rs_id', '') # Ensure rs_id is provided, even if empty
        }
        qr = conn.execute(query, parameters=params)
        if qr: del qr # Explicitly delete QueryResult
        print(f"Added variant: {variant_data['variant_id']}")
    except Exception as e:
        print(f"Error adding variant {variant_data.get('variant_id', 'UNKNOWN')}: {e}")
        raise

def add_sample(conn: kuzu.Connection, sample_data: Dict[str, Any]) -> None:
    """
    Adds a single sample node to the Kuzu database using parameterized queries.

    Args:
        conn: Active Kuzu connection.
        sample_data: Dictionary containing sample properties (sample_id).
                     Example: {'sample_id': 'SAMPLE_001'}
    """
    try:
        query = "CREATE (s:Sample {sample_id: $sample_id})"
        params = {"sample_id": sample_data['sample_id']}
        qr = conn.execute(query, parameters=params)
        if qr: del qr # Explicitly delete QueryResult
        print(f"Added sample: {sample_data['sample_id']}")
    except Exception as e:
        print(f"Error adding sample {sample_data.get('sample_id', 'UNKNOWN')}: {e}")
        raise

def link_variant_to_sample(conn: kuzu.Connection, sample_id: str, variant_id: str, properties: Dict[str, Any]) -> None:
    """
    Creates an 'ObservedIn' relationship between a Sample and a Variant using parameterized queries.

    Args:
        conn: Active Kuzu connection.
        sample_id: ID of the Sample node.
        variant_id: ID of the Variant node.
        properties: Dictionary of properties for the relationship (e.g., {'zygosity': 'HET'}).
    """
    try:
        if 'zygosity' not in properties:
            raise ValueError("Missing 'zygosity' in properties for ObservedIn relationship.")

        query = """
        MATCH (v:Variant {variant_id: $v_id}), (s:Sample {sample_id: $s_id})
        CREATE (v)-[r:ObservedIn {zygosity: $zygosity}]->(s)
        """
        params = {
            "v_id": variant_id,
            "s_id": sample_id,
            "zygosity": properties['zygosity']
        }
        qr = conn.execute(query, parameters=params)
        if qr: del qr # Explicitly delete QueryResult
        print(f"Linked sample {sample_id} to variant {variant_id} with properties: {properties}")
    except Exception as e:
        print(f"Error linking sample {sample_id} to variant {variant_id}: {e}")
        raise

def execute_query(conn: kuzu.Connection, cypher_query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Executes a Cypher query using the Kuzu connection and returns the results as a list of dictionaries.
    Parameters:
        - conn (kuzu.Connection): The database connection object to execute the query.
        - cypher_query (str): The Cypher query to execute.
        - params (Optional[Dict[str, Any]]): Parameters to include in the query, defaults to None.
    Returns:
        - List[Dict[str, Any]]: A list of dictionaries representing the query results.
    Processing Logic:
        - Handles both single and list results from the query execution, focusing on the first query result in case of a list.
        - Converts the query result to a Pandas DataFrame and then to a list of dictionaries.
        - Contains error handling for unexpected types and ensures cleanup of resources."""
    query_result_union: Optional[Union[kuzu.query_result.QueryResult, List[kuzu.query_result.QueryResult]]] = None
    try:
        print(f"Executing Cypher query: {cypher_query} with params: {params}")
        query_result_union = conn.execute(cypher_query, parameters=params if params else {})
        
        single_query_result: Optional[kuzu.query_result.QueryResult] = None
        if isinstance(query_result_union, list):
            if not query_result_union: # Empty list of results
                return []
            # Assuming we process the first QueryResult if a list is returned for some reason by Kuzu
            if isinstance(query_result_union[0], kuzu.query_result.QueryResult):
                single_query_result = query_result_union[0]
            else:
                raise TypeError(f"Kuzu conn.execute returned a list containing non-QueryResult: {query_result_union[0]}")
        elif isinstance(query_result_union, kuzu.query_result.QueryResult):
            single_query_result = query_result_union
        elif query_result_union is None: # Should not happen if execute always returns something or raises
             return [] # Or raise error
        else:
            raise TypeError(f"Kuzu conn.execute returned unexpected type: {type(query_result_union)}")

        if not single_query_result: # If somehow it's still None
            return []

        df = single_query_result.get_as_df()
        if not isinstance(df, pd.DataFrame):
            print(f"Warning: Expected Pandas DataFrame, but got {type(df)}.")
            raise TypeError(f"Conversion to DataFrame failed or returned unexpected type: {type(df)}.")
        results = df.to_dict(orient='records')
        print(f"Query returned {len(results)} results as dictionaries via DataFrame.")
        return results
    except Exception as e:
        print(f"Error executing Cypher query '{cypher_query}': {e}")
        raise
    finally:
        # Attempt to delete the union type or its components if they exist
        if query_result_union:
            if isinstance(query_result_union, list):
                for item in query_result_union:
                    if item: del item
            elif query_result_union: # It's a single QueryResult
                 del query_result_union
    return []  # Explicit return for linter

def get_variant_context(conn: Optional[kuzu.Connection], variant_ids: List[str]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetches graph context (e.g., samples, relationships) for a list of variant IDs from Kuzu.
    This function is intended to be called after a vector search in LanceDB provides relevant variant_ids.

    Args:
        conn: Optional active Kuzu connection. If None, uses the managed global connection.
        variant_ids: A list of variant IDs obtained from LanceDB vector search.

    Returns:
        A dictionary where keys are variant_ids and values are lists of dictionaries
        containing their graph context (e.g., associated samples and relationship properties).
        Example: {'rs123': [{'sample_id': 'S1', 'zygosity': 'HET'}]}}
    """
    if not variant_ids:
        return {}

    active_conn = conn
    if active_conn is None:
        active_conn = get_managed_kuzu_connection()
        if active_conn is None: # Should be caught by get_managed_kuzu_connection raising an error
            print("Error: Managed Kuzu connection is not available for get_variant_context.")
            # Return empty or raise, depending on desired strictness. For now, let's be strict.
            raise RuntimeError("Managed Kuzu connection could not be established for get_variant_context.")

    # Using UNWIND for efficient batch lookup if Kuzu supports it well for MATCH clauses.
    # Alternatively, loop and query one by one if UNWIND performance is not optimal for this pattern.
    query = """
    UNWIND $v_ids AS target_variant_id
    MATCH (v:Variant {variant_id: target_variant_id})-[o:ObservedIn]->(s:Sample)
    RETURN v.variant_id AS variant_id, s.sample_id AS sample_id, o.zygosity AS zygosity
    """
    params = {"v_ids": variant_ids}
    
    query_result_union_ctx: Optional[Union[kuzu.query_result.QueryResult, List[kuzu.query_result.QueryResult]]] = None
    try:
        query_result_union_ctx = active_conn.execute(query, parameters=params)
        
        single_query_result_ctx: Optional[kuzu.query_result.QueryResult] = None
        if isinstance(query_result_union_ctx, list):
            if not query_result_union_ctx: return {}
            if isinstance(query_result_union_ctx[0], kuzu.query_result.QueryResult):
                single_query_result_ctx = query_result_union_ctx[0]
            else:
                raise TypeError(f"Kuzu active_conn.execute returned list with non-QueryResult: {query_result_union_ctx[0]}")
        elif isinstance(query_result_union_ctx, kuzu.query_result.QueryResult):
            single_query_result_ctx = query_result_union_ctx
        elif query_result_union_ctx is None: 
            return {}
        else:
            raise TypeError(f"Kuzu active_conn.execute returned unexpected type: {type(query_result_union_ctx)}")

        if not single_query_result_ctx: return {}

        df = single_query_result_ctx.get_as_df()
        # Initialize context_map with all requested variant_ids to ensure they are all present in the output
        context_map: Dict[str, List[Dict[str, Any]]] = {v_id: [] for v_id in variant_ids}
        for record in df.to_dict(orient='records'):
            v_id = record['variant_id']
            # The v_id from the database record should always be in the initialized context_map if it was in variant_ids
            # If it's not (e.g. Kuzu query returned an ID not in the input list, though unlikely with UNWIND),
            # we could choose to ignore it or add it. Current logic assumes it will be present from initialization.
            context_map[v_id].append({'sample_id': record['sample_id'], 'zygosity': record['zygosity']})
        return context_map
    except Exception as e:
        print(f"Error in get_variant_context for variant_ids {variant_ids}: {e}")
        raise
    finally:
        if query_result_union_ctx:
            if isinstance(query_result_union_ctx, list):
                for item_ctx in query_result_union_ctx:
                    if item_ctx: del item_ctx
            elif query_result_union_ctx:
                del query_result_union_ctx
    return {}  # Explicit return for linter

def close_kuzu_connection(db_path: str = DEFAULT_KUZU_DB_PATH):
    """
    Closes the Kuzu database connection.

    Args:
        db_path: Path to the Kuzu database directory.
    """
    try:
        print(f"Closing Kuzu connection to: {db_path}")
        conn = get_kuzu_db_connection(db_path)
        conn.close()
        print("Kuzu connection closed successfully.")
    except Exception as e:
        print(f"Error closing Kuzu connection to {db_path}: {e}")
        raise

# Enhanced functions based on DECISION-001 specifications

def create_enhanced_schema(conn: kuzu.Connection) -> None:
    """
    Creates the enhanced schema for genomic data based on DECISION-001 specifications.
    
    Args:
        conn: Active Kuzu connection.
        
    Raises:
        Exception: If schema creation fails.
    """
    logger.info("Creating enhanced Kuzu schema based on DECISION-001 specifications...")
    
    try:
        for i, query in enumerate(ENHANCED_SCHEMA_QUERIES):
            logger.debug(f"Executing schema query {i+1}/{len(ENHANCED_SCHEMA_QUERIES)}: {query[:100]}...")
            qr = conn.execute(query)
            if qr: 
                del qr
        
        logger.info("Enhanced Kuzu schema created/verified successfully.")
        
    except Exception as e:
        logger.error(f"Error creating enhanced Kuzu schema: {e}")
        raise

def add_enhanced_sample(conn: kuzu.Connection, sample_data: Dict[str, Any]) -> None:
    """
    Adds a sample node using the enhanced schema.
    
    Args:
        conn: Active Kuzu connection.
        sample_data: Dictionary containing sample properties.
                    Required: id, name, type
                    Optional: metadata
    """
    try:
        query = """
        CREATE (s:Sample {
            id: $id,
            name: $name,
            type: $type,
            created_at: $created_at,
            metadata: $metadata
        })
        """
        
        params = {
            "id": sample_data['id'],
            "name": sample_data.get('name', sample_data['id']),
            "type": sample_data.get('type', 'unknown'),
            "created_at": datetime.now(),
            "metadata": sample_data.get('metadata', '')
        }
        
        qr = conn.execute(query, parameters=params)
        if qr: 
            del qr
            
        logger.info(f"Added enhanced sample: {sample_data['id']}")
        
    except Exception as e:
        logger.error(f"Error adding enhanced sample {sample_data.get('id', 'UNKNOWN')}: {e}")
        raise

def add_enhanced_variant(conn: kuzu.Connection, variant_data: Dict[str, Any]) -> None:
    """
    Adds a variant node using the enhanced schema.
    
    Args:
        conn: Active Kuzu connection.
        variant_data: Dictionary containing variant properties.
                     Required: id, chr, pos, ref, alt
                     Optional: variant_type, quality_score, filter_status, allele_frequency
    """
    try:
        query = """
        CREATE (v:Variant {
            id: $id,
            chr: $chr,
            pos: $pos,
            ref: $ref,
            alt: $alt,
            variant_type: $variant_type,
            quality_score: $quality_score,
            filter_status: $filter_status,
            allele_frequency: $allele_frequency,
            created_at: $created_at
        })
        """
        
        params = {
            "id": variant_data['id'],
            "chr": variant_data['chr'],
            "pos": variant_data['pos'],
            "ref": variant_data['ref'],
            "alt": variant_data['alt'],
            "variant_type": variant_data.get('variant_type', 'SNV'),
            "quality_score": variant_data.get('quality_score', 0.0),
            "filter_status": variant_data.get('filter_status', 'UNKNOWN'),
            "allele_frequency": variant_data.get('allele_frequency', 0.0),
            "created_at": datetime.now()
        }
        
        qr = conn.execute(query, parameters=params)
        if qr: 
            del qr
            
        logger.info(f"Added enhanced variant: {variant_data['id']}")
        
    except Exception as e:
        logger.error(f"Error adding enhanced variant {variant_data.get('id', 'UNKNOWN')}: {e}")
        raise

def add_gene(conn: kuzu.Connection, gene_data: Dict[str, Any]) -> None:
    """
    Adds a gene node to the database.
    
    Args:
        conn: Active Kuzu connection.
        gene_data: Dictionary containing gene properties.
                  Required: id, symbol, name, chromosome
                  Optional: start_pos, end_pos, biotype
    """
    try:
        query = """
        CREATE (g:Gene {
            id: $id,
            symbol: $symbol,
            name: $name,
            chromosome: $chromosome,
            start_pos: $start_pos,
            end_pos: $end_pos,
            biotype: $biotype
        })
        """
        
        params = {
            "id": gene_data['id'],
            "symbol": gene_data['symbol'],
            "name": gene_data.get('name', gene_data['symbol']),
            "chromosome": gene_data['chromosome'],
            "start_pos": gene_data.get('start_pos', 0),
            "end_pos": gene_data.get('end_pos', 0),
            "biotype": gene_data.get('biotype', 'protein_coding')
        }
        
        qr = conn.execute(query, parameters=params)
        if qr: 
            del qr
            
        logger.info(f"Added gene: {gene_data['symbol']} ({gene_data['id']})")
        
    except Exception as e:
        logger.error(f"Error adding gene {gene_data.get('symbol', 'UNKNOWN')}: {e}")
        raise

def add_analysis(conn: kuzu.Connection, analysis_data: Dict[str, Any]) -> None:
    """
    Adds an analysis node to the database.
    
    Args:
        conn: Active Kuzu connection.
        analysis_data: Dictionary containing analysis properties.
                      Required: id, type, summary
                      Optional: confidence_score
    """
    try:
        query = """
        CREATE (a:Analysis {
            id: $id,
            type: $type,
            summary: $summary,
            confidence_score: $confidence_score,
            created_at: $created_at
        })
        """
        
        params = {
            "id": analysis_data['id'],
            "type": analysis_data['type'],
            "summary": analysis_data['summary'],
            "confidence_score": analysis_data.get('confidence_score', 0.0),
            "created_at": datetime.now()
        }
        
        qr = conn.execute(query, parameters=params)
        if qr: 
            del qr
            
        logger.info(f"Added analysis: {analysis_data['id']}")
        
    except Exception as e:
        logger.error(f"Error adding analysis {analysis_data.get('id', 'UNKNOWN')}: {e}")
        raise

def create_has_variant_relationship(
    conn: kuzu.Connection, 
    sample_id: str, 
    variant_id: str, 
    relationship_data: Dict[str, Any]
) -> None:
    """
    Creates a HasVariant relationship between a Sample and a Variant.
    
    Args:
        conn: Active Kuzu connection.
        sample_id: ID of the Sample node.
        variant_id: ID of the Variant node.
        relationship_data: Dictionary containing relationship properties.
                          Required: genotype
                          Optional: quality, depth, allele_depth
    """
    try:
        query = """
        MATCH (s:Sample {id: $sample_id}), (v:Variant {id: $variant_id})
        CREATE (s)-[r:HasVariant {
            genotype: $genotype,
            quality: $quality,
            depth: $depth,
            allele_depth: $allele_depth,
            created_at: $created_at
        }]->(v)
        """
        
        params = {
            "sample_id": sample_id,
            "variant_id": variant_id,
            "genotype": relationship_data['genotype'],
            "quality": relationship_data.get('quality', 0.0),
            "depth": relationship_data.get('depth', 0),
            "allele_depth": relationship_data.get('allele_depth', ''),
            "created_at": datetime.now()
        }
        
        qr = conn.execute(query, parameters=params)
        if qr: 
            del qr
            
        logger.info(f"Created HasVariant relationship: {sample_id} -> {variant_id}")
        
    except Exception as e:
        logger.error(f"Error creating HasVariant relationship {sample_id} -> {variant_id}: {e}")
        raise

def create_located_in_relationship(
    conn: kuzu.Connection, 
    variant_id: str, 
    gene_id: str, 
    relationship_data: Dict[str, Any]
) -> None:
    """
    Creates a LocatedIn relationship between a Variant and a Gene.
    
    Args:
        conn: Active Kuzu connection.
        variant_id: ID of the Variant node.
        gene_id: ID of the Gene node.
        relationship_data: Dictionary containing relationship properties.
                          Optional: impact, consequence, amino_acid_change, codon_change
    """
    try:
        query = """
        MATCH (v:Variant {id: $variant_id}), (g:Gene {id: $gene_id})
        CREATE (v)-[r:LocatedIn {
            impact: $impact,
            consequence: $consequence,
            amino_acid_change: $amino_acid_change,
            codon_change: $codon_change
        }]->(g)
        """
        
        params = {
            "variant_id": variant_id,
            "gene_id": gene_id,
            "impact": relationship_data.get('impact', ''),
            "consequence": relationship_data.get('consequence', ''),
            "amino_acid_change": relationship_data.get('amino_acid_change', ''),
            "codon_change": relationship_data.get('codon_change', '')
        }
        
        qr = conn.execute(query, parameters=params)
        if qr: 
            del qr
            
        logger.info(f"Created LocatedIn relationship: {variant_id} -> {gene_id}")
        
    except Exception as e:
        logger.error(f"Error creating LocatedIn relationship {variant_id} -> {gene_id}: {e}")
        raise

def batch_add_genomic_data(
    conn: kuzu.Connection,
    samples: List[Dict[str, Any]],
    variants: List[Dict[str, Any]],
    genes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    batch_size: int = 1000
) -> Dict[str, int]:
    """
    Batch add genomic data with optimized performance.
    Target: <500ms for complex operations as per DECISION-001.
    
    Args:
        conn: Active Kuzu connection.
        samples: List of sample data dictionaries.
        variants: List of variant data dictionaries.
        genes: List of gene data dictionaries.
        relationships: List of relationship data dictionaries.
        batch_size: Number of items to process in each batch.
        
    Returns:
        Dictionary with counts of added items.
    """
    start_time = time.time()
    logger.info(f"Starting batch genomic data insertion: {len(samples)} samples, {len(variants)} variants, {len(genes)} genes, {len(relationships)} relationships")
    
    counts = {"samples": 0, "variants": 0, "genes": 0, "relationships": 0}
    
    try:
        # Add samples in batches
        for i in range(0, len(samples), batch_size):
            batch = samples[i:i + batch_size]
            for sample_data in batch:
                add_enhanced_sample(conn, sample_data)
                counts["samples"] += 1
        
        # Add variants in batches
        for i in range(0, len(variants), batch_size):
            batch = variants[i:i + batch_size]
            for variant_data in batch:
                add_enhanced_variant(conn, variant_data)
                counts["variants"] += 1
        
        # Add genes in batches
        for i in range(0, len(genes), batch_size):
            batch = genes[i:i + batch_size]
            for gene_data in batch:
                add_gene(conn, gene_data)
                counts["genes"] += 1
        
        # Add relationships in batches
        for i in range(0, len(relationships), batch_size):
            batch = relationships[i:i + batch_size]
            for rel_data in batch:
                if rel_data['type'] == 'HasVariant':
                    create_has_variant_relationship(
                        conn, 
                        rel_data['sample_id'], 
                        rel_data['variant_id'], 
                        rel_data['properties']
                    )
                elif rel_data['type'] == 'LocatedIn':
                    create_located_in_relationship(
                        conn, 
                        rel_data['variant_id'], 
                        rel_data['gene_id'], 
                        rel_data['properties']
                    )
                counts["relationships"] += 1
        
        elapsed_time = time.time() - start_time
        logger.info(f"Batch genomic data insertion completed in {elapsed_time:.2f}s: {counts}")
        
        return counts
        
    except Exception as e:
        logger.error(f"Error in batch genomic data insertion: {e}")
        raise

def find_sample_variants(conn: kuzu.Connection, sample_id: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Find all variants for a specific sample with relationship details.
    
    Args:
        conn: Active Kuzu connection.
        sample_id: ID of the sample.
        limit: Maximum number of variants to return.
        
    Returns:
        List of variant dictionaries with relationship properties.
    """
    try:
        query = """
        MATCH (s:Sample {id: $sample_id})-[r:HasVariant]->(v:Variant)
        RETURN v.id as variant_id, v.chr as chromosome, v.pos as position,
               v.ref as reference, v.alt as alternate, v.quality_score,
               r.genotype, r.quality as relationship_quality, r.depth
        LIMIT $limit
        """
        
        params = {"sample_id": sample_id, "limit": limit}
        results = execute_query(conn, query, params)
        
        logger.info(f"Found {len(results)} variants for sample {sample_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error finding variants for sample {sample_id}: {e}")
        raise

def find_variant_genes(conn: kuzu.Connection, variant_id: str) -> List[Dict[str, Any]]:
    """
    Find all genes associated with a specific variant.
    
    Args:
        conn: Active Kuzu connection.
        variant_id: ID of the variant.
        
    Returns:
        List of gene dictionaries with relationship properties.
    """
    try:
        query = """
        MATCH (v:Variant {id: $variant_id})-[r:LocatedIn]->(g:Gene)
        RETURN g.id as gene_id, g.symbol, g.name, g.chromosome,
               r.impact, r.consequence, r.amino_acid_change, r.codon_change
        """
        
        params = {"variant_id": variant_id}
        results = execute_query(conn, query, params)
        
        logger.info(f"Found {len(results)} genes for variant {variant_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error finding genes for variant {variant_id}: {e}")
        raise

def find_similar_samples(conn: kuzu.Connection, sample_id: str, min_shared_variants: int = 5) -> List[Dict[str, Any]]:
    """
    Find samples that share variants with the given sample.
    
    Args:
        conn: Active Kuzu connection.
        sample_id: ID of the reference sample.
        min_shared_variants: Minimum number of shared variants.
        
    Returns:
        List of similar samples with shared variant counts.
    """
    try:
        query = """
        MATCH (s1:Sample {id: $sample_id})-[:HasVariant]->(v:Variant)<-[:HasVariant]-(s2:Sample)
        WHERE s1.id <> s2.id
        WITH s2, COUNT(v) as shared_variants
        WHERE shared_variants >= $min_shared_variants
        RETURN s2.id as sample_id, s2.name, s2.type, shared_variants
        ORDER BY shared_variants DESC
        """
        
        params = {"sample_id": sample_id, "min_shared_variants": min_shared_variants}
        results = execute_query(conn, query, params)
        
        logger.info(f"Found {len(results)} similar samples for {sample_id}")
        return results
        
    except Exception as e:
        logger.error(f"Error finding similar samples for {sample_id}: {e}")
        raise

def get_genomic_statistics(conn: kuzu.Connection) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the genomic data in the graph.
    
    Args:
        conn: Active Kuzu connection.
        
    Returns:
        Dictionary containing various statistics.
    """
    try:
        stats = {}
        
        # Count nodes
        node_counts = {}
        for node_type in ['Sample', 'Variant', 'Gene', 'Analysis']:
            query = f"MATCH (n:{node_type}) RETURN COUNT(n) as count"
            result = execute_query(conn, query)
            node_counts[node_type.lower()] = result[0]['count'] if result else 0
        
        stats['node_counts'] = node_counts
        
        # Count relationships
        rel_counts = {}
        for rel_type in ['HasVariant', 'LocatedIn', 'AnalyzedBy', 'SimilarTo']:
            query = f"MATCH ()-[r:{rel_type}]->() RETURN COUNT(r) as count"
            result = execute_query(conn, query)
            rel_counts[rel_type] = result[0]['count'] if result else 0
        
        stats['relationship_counts'] = rel_counts
        
        # Variant distribution by chromosome
        query = """
        MATCH (v:Variant)
        RETURN v.chr as chromosome, COUNT(v) as variant_count
        ORDER BY variant_count DESC
        LIMIT 25
        """
        chrom_dist = execute_query(conn, query)
        stats['chromosome_distribution'] = chrom_dist
        
        # Sample variant counts
        query = """
        MATCH (s:Sample)-[:HasVariant]->(v:Variant)
        WITH s, COUNT(v) as variant_count
        RETURN AVG(variant_count) as avg_variants_per_sample,
               MIN(variant_count) as min_variants,
               MAX(variant_count) as max_variants
        """
        sample_stats = execute_query(conn, query)
        stats['sample_statistics'] = sample_stats[0] if sample_stats else {}
        
        logger.info(f"Generated genomic statistics: {stats['node_counts']}")
        return stats
        
    except Exception as e:
        logger.error(f"Error generating genomic statistics: {e}")
        return {"error": str(e)}

if __name__ == '__main__':
    # Example Usage (illustrative, to be updated for get_variant_context)
    # db_connection = None
    # try:
    #     print("Attempting to connect to Kuzu and set up schema...")
    #     db_connection = get_kuzu_db_connection()
    #     create_schema(db_connection)
    #
    #     # Add some data (variant, sample, link)
    #     var_data1 = {'variant_id': 'rs123', 'chrom': '1', 'pos': 1000, 'ref': 'A', 'alt': 'T'}
    #     var_data2 = {'variant_id': 'rs456', 'chrom': '2', 'pos': 2000, 'ref': 'C', 'alt': 'G'}
    #     samp_data1 = {'sample_id': 'SAMPLE001'}
    #     samp_data2 = {'sample_id': 'SAMPLE002'}
    #     add_variant(db_connection, var_data1)
    #     add_variant(db_connection, var_data2)
    #     add_sample(db_connection, samp_data1)
    #     add_sample(db_connection, samp_data2)
    #     link_variant_to_sample(db_connection, 'SAMPLE001', 'rs123', {'zygosity': 'HET'})
    #     link_variant_to_sample(db_connection, 'SAMPLE001', 'rs456', {'zygosity': 'HOM'})
    #     link_variant_to_sample(db_connection, 'SAMPLE002', 'rs123', {'zygosity': 'HOM'})
    #
    #     print("\nQuerying all ObservedIn relationships...")
    #     all_links = execute_query(db_connection, "MATCH (v:Variant)-[o:ObservedIn]->(s:Sample) RETURN v.variant_id, s.sample_id, o.zygosity")
    #     for link in all_links:
    #         print(f"  Found Link: {link}")
    #
    #     # Simulate results from LanceDB vector search
    #     lancedb_results_variant_ids = ['rs123', 'rs456'] 
    #     print(f"\nFetching Kuzu context for variant IDs: {lancedb_results_variant_ids}")
    #     variant_contexts = get_variant_context(db_connection, lancedb_results_variant_ids)
    #     for v_id, context in variant_contexts.items():
    #         print(f"  Context for {v_id}: {context}")
    #
    # except NotImplementedError as nie:
    #     print(f"Feature not implemented: {nie}")
    # except Exception as e:
    #     print(f"An error occurred during Kuzu example usage: {e}")
    # finally:
    #     if db_connection:
    #         print("\nKuzu operations example finished.")
    pass 