"""
CLI commands for SAMspec compliance validation.
"""

import click
import json
import sys
from pathlib import Path
from typing import Optional

from ..samspec_compliance import validate_vcf_samspec_compliance, ComplianceLevel


@click.group()
def samspec():
    """SAMspec compliance validation commands."""
    pass


@samspec.command()
@click.argument('vcf_file', type=click.Path(exists=True))
@click.option('--output', '-o', type=click.Path(), help='Output file for compliance report')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text', 
              help='Output format (text or json)')
@click.option('--strict', is_flag=True, help='Treat warnings as failures')
@click.option('--quiet', '-q', is_flag=True, help='Only show summary')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed violation information')
def validate(vcf_file: str, output: Optional[str], format: str, strict: bool, 
             quiet: bool, verbose: bool) -> None:
    """
    Validate VCF file for SAMspec compliance.
    
    Validates the specified VCF file against the SAM/VCF specification standards
    and reports any compliance violations found.
    
    Examples:
        vcf-agent samspec validate sample.vcf
        vcf-agent samspec validate sample.vcf --format json --output report.json
        vcf-agent samspec validate sample.vcf --strict --verbose
    """
    try:
        # Perform validation
        click.echo(f"Validating VCF file: {vcf_file}")
        report = validate_vcf_samspec_compliance(vcf_file)
        
        # Generate output
        if format == 'json':
            output_content = json.dumps(report.to_dict(), indent=2)
        else:
            output_content = _format_text_report(report, verbose, quiet)
        
        # Write output
        if output:
            Path(output).write_text(output_content)
            click.echo(f"Report written to: {output}")
        else:
            click.echo(output_content)
        
        # Determine exit code
        exit_code = 0
        if report.critical_count > 0:
            exit_code = 1
        elif strict and report.warning_count > 0:
            exit_code = 1
        
        if exit_code != 0:
            click.echo(f"\nValidation failed with {report.critical_count} critical "
                      f"and {report.warning_count} warning violations.", err=True)
        else:
            click.echo(f"\nValidation passed! File is SAMspec compliant.")
        
        sys.exit(exit_code)
        
    except Exception as e:
        click.echo(f"Error during validation: {str(e)}", err=True)
        sys.exit(1)


@samspec.command()
@click.argument('vcf_files', nargs=-1, type=click.Path(exists=True), required=True)
@click.option('--output-dir', '-d', type=click.Path(), help='Output directory for reports')
@click.option('--format', '-f', type=click.Choice(['text', 'json']), default='text',
              help='Output format (text or json)')
@click.option('--strict', is_flag=True, help='Treat warnings as failures')
@click.option('--summary', is_flag=True, help='Show summary report for all files')
def batch_validate(vcf_files: tuple, output_dir: Optional[str], format: str, 
                   strict: bool, summary: bool) -> None:
    """
    Validate multiple VCF files for SAMspec compliance.
    
    Validates multiple VCF files and optionally generates individual reports
    and a summary report.
    
    Examples:
        vcf-agent samspec batch-validate *.vcf
        vcf-agent samspec batch-validate file1.vcf file2.vcf --output-dir reports/
        vcf-agent samspec batch-validate *.vcf --summary --strict
    """
    try:
        results = []
        failed_files = []
        
        # Create output directory if specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
        
        # Validate each file
        for vcf_file in vcf_files:
            click.echo(f"Validating: {vcf_file}")
            
            try:
                report = validate_vcf_samspec_compliance(vcf_file)
                results.append(report)
                
                # Check if file failed validation
                file_failed = report.critical_count > 0
                if strict and report.warning_count > 0:
                    file_failed = True
                
                if file_failed:
                    failed_files.append(vcf_file)
                
                # Generate individual report if output directory specified
                if output_dir:
                    file_stem = Path(vcf_file).stem
                    if format == 'json':
                        report_file = output_path / f"{file_stem}_compliance.json"
                        content = json.dumps(report.to_dict(), indent=2)
                    else:
                        report_file = output_path / f"{file_stem}_compliance.txt"
                        content = _format_text_report(report, verbose=True, quiet=False)
                    
                    report_file.write_text(content)
                    click.echo(f"  Report: {report_file}")
                
                # Show brief status
                status = "PASS" if not file_failed else "FAIL"
                click.echo(f"  Status: {status} ({report.critical_count} critical, "
                          f"{report.warning_count} warnings)")
                
            except Exception as e:
                click.echo(f"  Error: {str(e)}", err=True)
                failed_files.append(vcf_file)
        
        # Generate summary report
        if summary and results:
            summary_content = _format_summary_report(results, strict)
            
            if output_dir:
                summary_file = output_path / f"compliance_summary.{format}"
                if format == 'json':
                    summary_data = {
                        "total_files": len(results),
                        "passed_files": len(results) - len(failed_files),
                        "failed_files": len(failed_files),
                        "strict_mode": strict,
                        "files": [report.to_dict() for report in results]
                    }
                    summary_file.write_text(json.dumps(summary_data, indent=2))
                else:
                    summary_file.write_text(summary_content)
                click.echo(f"\nSummary report: {summary_file}")
            else:
                click.echo("\n" + summary_content)
        
        # Final status
        total_files = len(vcf_files)
        passed_files = total_files - len(failed_files)
        
        click.echo(f"\nBatch validation complete:")
        click.echo(f"  Total files: {total_files}")
        click.echo(f"  Passed: {passed_files}")
        click.echo(f"  Failed: {len(failed_files)}")
        
        if failed_files:
            click.echo(f"\nFailed files:")
            for file in failed_files:
                click.echo(f"  - {file}")
            sys.exit(1)
        else:
            click.echo(f"\nAll files passed SAMspec compliance validation!")
            sys.exit(0)
            
    except Exception as e:
        click.echo(f"Error during batch validation: {str(e)}", err=True)
        sys.exit(1)


@samspec.command()
@click.argument('vcf_file', type=click.Path(exists=True))
@click.option('--rule-id', help='Show details for specific rule ID')
@click.option('--level', type=click.Choice(['critical', 'warning', 'info']),
              help='Filter violations by level')
def explain(vcf_file: str, rule_id: Optional[str], level: Optional[str]) -> None:
    """
    Explain SAMspec compliance violations in detail.
    
    Provides detailed explanations of compliance violations found in a VCF file,
    with optional filtering by rule ID or violation level.
    
    Examples:
        vcf-agent samspec explain sample.vcf
        vcf-agent samspec explain sample.vcf --rule-id MISSING_FILEFORMAT
        vcf-agent samspec explain sample.vcf --level critical
    """
    try:
        report = validate_vcf_samspec_compliance(vcf_file)
        
        # Filter violations if requested
        violations = report.violations
        if rule_id:
            violations = [v for v in violations if v.rule_id == rule_id]
        if level:
            violations = [v for v in violations if v.level.value == level]
        
        if not violations:
            if rule_id or level:
                click.echo("No violations found matching the specified criteria.")
            else:
                click.echo("No compliance violations found. File is SAMspec compliant!")
            return
        
        click.echo(f"SAMspec Compliance Violations for: {vcf_file}")
        click.echo("=" * 60)
        
        for i, violation in enumerate(violations, 1):
            click.echo(f"\n{i}. {violation.rule_id} ({violation.level.value.upper()})")
            click.echo(f"   Message: {violation.message}")
            
            if violation.line_number:
                click.echo(f"   Line: {violation.line_number}")
            if violation.field:
                click.echo(f"   Field: {violation.field}")
            if violation.value:
                click.echo(f"   Value: {violation.value}")
            if violation.suggestion:
                click.echo(f"   Suggestion: {violation.suggestion}")
        
        click.echo(f"\nTotal violations: {len(violations)}")
        
    except Exception as e:
        click.echo(f"Error during explanation: {str(e)}", err=True)
        sys.exit(1)


def _format_text_report(report, verbose: bool = False, quiet: bool = False) -> str:
    """Format compliance report as text."""
    lines = []
    
    if not quiet:
        lines.append("SAMspec Compliance Report")
        lines.append("=" * 50)
        lines.append(f"File: {report.file_path}")
        lines.append(f"VCF Version: {report.vcf_version or 'Unknown'}")
        lines.append(f"Compliant: {'Yes' if report.is_compliant else 'No'}")
        lines.append("")
    
    # Summary
    lines.append("Violation Summary:")
    lines.append(f"  Critical: {report.critical_count}")
    lines.append(f"  Warnings: {report.warning_count}")
    lines.append(f"  Info: {report.info_count}")
    lines.append(f"  Total: {report.total_violations}")
    
    # Detailed violations
    if verbose and report.violations:
        lines.append("\nDetailed Violations:")
        lines.append("-" * 30)
        
        for i, violation in enumerate(report.violations, 1):
            lines.append(f"\n{i}. {violation.rule_id} ({violation.level.value.upper()})")
            lines.append(f"   {violation.message}")
            
            if violation.line_number:
                lines.append(f"   Line: {violation.line_number}")
            if violation.field:
                lines.append(f"   Field: {violation.field}")
            if violation.suggestion:
                lines.append(f"   Suggestion: {violation.suggestion}")
    
    return "\n".join(lines)


def _format_summary_report(reports, strict: bool = False) -> str:
    """Format summary report for multiple files."""
    lines = []
    lines.append("SAMspec Compliance Summary Report")
    lines.append("=" * 50)
    
    total_files = len(reports)
    compliant_files = sum(1 for r in reports if r.is_compliant)
    failed_files = total_files - compliant_files
    
    if strict:
        # In strict mode, warnings count as failures
        failed_files = sum(1 for r in reports if r.critical_count > 0 or r.warning_count > 0)
        compliant_files = total_files - failed_files
    
    lines.append(f"Total Files: {total_files}")
    lines.append(f"Compliant: {compliant_files}")
    lines.append(f"Non-compliant: {failed_files}")
    lines.append(f"Strict Mode: {'Yes' if strict else 'No'}")
    lines.append("")
    
    # Per-file summary
    lines.append("Per-File Results:")
    lines.append("-" * 30)
    
    for report in reports:
        file_name = Path(report.file_path).name
        status = "PASS"
        
        if report.critical_count > 0:
            status = "FAIL"
        elif strict and report.warning_count > 0:
            status = "FAIL"
        
        lines.append(f"{file_name}: {status} "
                    f"(C:{report.critical_count}, W:{report.warning_count}, I:{report.info_count})")
    
    # Overall statistics
    total_critical = sum(r.critical_count for r in reports)
    total_warnings = sum(r.warning_count for r in reports)
    total_info = sum(r.info_count for r in reports)
    
    lines.append("")
    lines.append("Overall Statistics:")
    lines.append(f"  Total Critical Violations: {total_critical}")
    lines.append(f"  Total Warnings: {total_warnings}")
    lines.append(f"  Total Info: {total_info}")
    
    return "\n".join(lines) 