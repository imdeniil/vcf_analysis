"""
Python wrapper for GATK ValidateVariants command.
Uses subprocess for robust execution and output/error capture.
Extensible for additional GATK commands.
"""

import subprocess
from typing import List, Optional, Tuple

def run_gatk_command(command: List[str], input_data: Optional[bytes] = None) -> Tuple[int, str, str]:
    """
    Run a GATK command and capture output and errors.

    Args:
        command (List[str]): List of command arguments (e.g., ["ValidateVariants", "-V", "file.vcf"])
        input_data (Optional[bytes]): Optional bytes to pass to stdin.

    Returns:
        Tuple[int, str, str]: (returncode, stdout, stderr)

    Raises:
        FileNotFoundError: If GATK is not installed.

    Example:
        >>> run_gatk_command(["ValidateVariants", "-V", "sample_data/HG00098.vcf.gz", "-R", "ref.fasta"])
        (0, '...', '')
    """
    full_cmd = ["gatk"] + command
    try:
        result = subprocess.run(
            full_cmd,
            input=input_data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False
        )
        return result.returncode, result.stdout.decode(), result.stderr.decode()
    except FileNotFoundError as e:
        raise FileNotFoundError("GATK is not installed or not in PATH") from e

def gatk_validatevariants(args: List[str], input_data: Optional[bytes] = None) -> Tuple[int, str, str]:
    """
    Run GATK ValidateVariants with the given arguments.

    Args:
        args (List[str]): Arguments for GATK ValidateVariants.
        input_data (Optional[bytes]): Optional bytes to pass to stdin.

    Returns:
        Tuple[int, str, str]: (returncode, stdout, stderr)

    Example:
        >>> gatk_validatevariants(["-V", "sample_data/HG00098.vcf.gz", "-R", "ref.fasta"])
        (0, '...', '')
    """
    return run_gatk_command(["ValidateVariants"] + args, input_data) 