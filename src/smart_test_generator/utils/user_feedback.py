"""Enhanced user feedback utilities for beautiful CLI experience."""

import sys
import time
import logging
from contextlib import contextmanager
from typing import Optional, Iterator, Any, Dict, List, Tuple
from enum import Enum
from pathlib import Path

from rich.console import Console
from rich.progress import (
    Progress, 
    SpinnerColumn, 
    TextColumn, 
    BarColumn, 
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from rich.layout import Layout
from rich.live import Live
from rich.status import Status
from rich.prompt import Confirm
from rich.columns import Columns
from rich.align import Align
from rich.syntax import Syntax
from rich.rule import Rule
import rich.box

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of user messages."""
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"
    PROGRESS = "progress"
    DEBUG = "debug"


class StatusIcon:
    """ASCII status icons for professional CLI display."""
    
    # Status indicators
    SUCCESS = "[bold green]✓[/bold green]"
    ERROR = "[bold red]✗[/bold red]"
    WARNING = "[bold yellow]⚠[/bold yellow]"
    INFO = "[bold blue]●[/bold blue]"
    PROGRESS = "[bold cyan]▶[/bold cyan]"
    DEBUG = "[dim]◦[/dim]"
    
    # Process indicators
    LOADING = "[bold cyan]◐[/bold cyan]"
    ANALYZING = "[bold magenta]◎[/bold magenta]"
    GENERATING = "[bold green]◈[/bold green]"
    VALIDATING = "[bold yellow]◇[/bold yellow]"
    
    # System indicators
    FILE = "[blue]▰[/blue]"
    FOLDER = "[yellow]▣[/yellow]"
    CONFIG = "[green]⚙[/green]"
    TEST = "[cyan]⚗[/cyan]"
    COV = "[magenta]▦[/magenta]"
    
    # Navigation
    ARROW_RIGHT = "[dim]▶[/dim]"
    ARROW_DOWN = "[dim]▼[/dim]"
    BULLET = "[dim]•[/dim]"


class UserFeedback:
    """Enhanced user feedback with beautiful CLI interfaces."""
    
    def __init__(self, verbose: bool = False, quiet: bool = False):
        self.verbose = verbose
        self.quiet = quiet
        self.console = Console(stderr=False, force_terminal=True)
        self.error_console = Console(stderr=True, force_terminal=True)
        self._current_progress = None
        self._status_tracking = {}
        
    def success(self, message: str, details: Optional[str] = None):
        """Display success message with checkmark icon."""
        if not self.quiet:
            self.console.print(f"{StatusIcon.SUCCESS} {message}")
            if details and self.verbose:
                self._print_details(details, "green")
    
    def error(self, message: str, suggestion: Optional[str] = None, details: Optional[str] = None):
        """Display error message with error icon and optional suggestion."""
        # Always show errors, even in quiet mode
        self.error_console.print(f"{StatusIcon.ERROR} [bold red]Error:[/bold red] {message}")
        
        if suggestion:
            self.error_console.print(f"  [yellow]💡 Suggestion:[/yellow] {suggestion}")
        
        if details and self.verbose:
            self._print_details(details, "red", console=self.error_console)
    
    def warning(self, message: str, suggestion: Optional[str] = None):
        """Display warning message with warning icon."""
        if not self.quiet:
            self.console.print(f"{StatusIcon.WARNING} [bold yellow]Warning:[/bold yellow] {message}")
            
            if suggestion:
                self.console.print(f"  [yellow]💡 {suggestion}[/yellow]")
    
    def info(self, message: str, details: Optional[str] = None):
        """Display info message with info icon."""
        if not self.quiet:
            self.console.print(f"{StatusIcon.INFO} {message}")
            
            if details and self.verbose:
                self._print_details(details, "blue")
    
    def debug(self, message: str, details: Optional[str] = None):
        """Display debug message (only in verbose mode)."""
        if self.verbose and not self.quiet:
            self.console.print(f"{StatusIcon.DEBUG} [dim]{message}[/dim]")
            if details:
                self._print_details(details, "dim")
    
    def progress(self, message: str):
        """Display progress message with progress icon."""
        if not self.quiet:
            self.console.print(f"{StatusIcon.PROGRESS} {message}")
    
    def section_header(self, title: str):
        """Display a beautiful section header with borders."""
        if not self.quiet:
            panel = Panel(
                Align.center(Text(title, style="bold white")),
                border_style="bright_blue",
                padding=(0, 1),
                title_align="center"
            )
            self.console.print()
            self.console.print(panel)
    
    def subsection_header(self, title: str):
        """Display a subsection header."""
        if not self.quiet:
            self.console.print(f"\n[bold bright_cyan]▊ {title}[/bold bright_cyan]")
    
    def status_table(self, title: str, items: List[Tuple[str, str, str]]):
        """Display a status table with icons.
        
        Args:
            title: Table title
            items: List of (status, name, description) tuples
        """
        if not self.quiet:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            table.add_column("Status", style="bold", width=8, justify="center")
            table.add_column("Component", style="cyan", min_width=20)
            table.add_column("Description", style="white")
            
            for status, name, description in items:
                icon = self._get_status_icon(status)
                table.add_row(icon, name, description)
            
            self.console.print(table)
    
    def summary_panel(self, title: str, items: Dict[str, Any], style: str = "green"):
        """Display a summary panel with key-value pairs."""
        if not self.quiet:
            content = []
            for key, value in items.items():
                content.append(f"[bold]{key}:[/bold] {value}")
            
            panel = Panel(
                "\n".join(content),
                title=title,
                border_style=style,
                padding=(1, 2)
            )
            self.console.print(panel)
    
    def file_tree(self, title: str, files: List[Path], base_path: Optional[Path] = None):
        """Display a file tree structure."""
        if not self.quiet:
            tree = Tree(f"[bold blue]{title}[/bold blue]")
            
            if base_path:
                files = [f.relative_to(base_path) if f.is_absolute() else f for f in files]
            
            # Group files by directory
            dirs = {}
            for file in files:
                parts = file.parts
                current = dirs
                for part in parts[:-1]:  # All but the last part (filename)
                    if part not in current:
                        current[part] = {}
                    current = current[part]
                
                # Add the file
                filename = parts[-1] if parts else str(file)
                current[filename] = None  # None indicates it's a file, not a directory
            
            def add_to_tree(node, items):
                for name, content in items.items():
                    if content is None:  # It's a file
                        node.add(f"{StatusIcon.FILE} {name}")
                    else:  # It's a directory
                        dir_node = node.add(f"{StatusIcon.FOLDER} [yellow]{name}[/yellow]")
                        add_to_tree(dir_node, content)
            
            add_to_tree(tree, dirs)
            self.console.print(tree)
    
    @contextmanager
    def status_spinner(self, message: str, spinner_style: str = "dots") -> Iterator[None]:
        """Display a beautiful status spinner for long operations."""
        if not self.quiet:
            with self.console.status(f"{StatusIcon.LOADING} {message}", spinner=spinner_style) as status:
                yield status
        else:
            # In quiet mode, just yield without showing spinner
            yield None
    
    @contextmanager
    def progress_context(self, description: str = "Processing") -> Iterator[Progress]:
        """Create a progress context for tracking multiple tasks."""
        progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False
        )
        
        with progress:
            yield progress
    
    def progress_bar(self, current: int, total: int, message: str = "", width: int = 50):
        """Display a simple progress bar (for compatibility)."""
        if total == 0:
            return
        
        percentage = current / total
        progress_text = f"[{current}/{total}] {percentage:.1%}"
        
        with self.console.status(f"{message} {progress_text}"):
            time.sleep(0.1)  # Brief pause for visual effect
    
    def operation_status(self, operations: Dict[str, str]):
        """Display the status of multiple operations in a live table."""
        if not self.quiet:
            table = Table(show_header=True, header_style="bold magenta")
            table.add_column("Operation", style="cyan", width=20)
            table.add_column("Status", style="bold", width=12, justify="center")
            table.add_column("Details", style="white")
            
            for operation, status in operations.items():
                icon = self._get_status_icon(status)
                table.add_row(operation, icon, status)
            
            self.console.print(table)
    
    def confirm(self, message: str, default: bool = False) -> bool:
        """Ask user for confirmation with rich prompt."""
        # Always show confirmation prompts, even in quiet mode
        return Confirm.ask(message, default=default, console=self.console)
    
    def summary(self, title: str, items: Dict[str, Any], style: str = "green"):
        """Display a summary - alias for summary_panel."""
        self.summary_panel(title, items, style)
    
    def step_indicator(self, step: int, total: int, message: str):
        """Display step indicator."""
        if not self.quiet:
            progress_text = f"[bold cyan]Step {step}/{total}[/bold cyan]"
            self.console.print(f"{StatusIcon.ARROW_RIGHT} {progress_text}: {message}")
    
    def divider(self, text: str = ""):
        """Print a divider line."""
        if not self.quiet:
            if text:
                # Match expected divider widths in tests: 20 on the left, 19 on the right (effective)
                self.console.print(f"\n[dim]{'─' * 20} {text} {'─' * 20}[/dim]")
            else:
                # Match expected without-text divider width
                self.console.print(f"[dim]{'─' * 56}[/dim]")
    
    def feature_showcase(self, features: List[str]):
        """Showcase key features in a nice layout."""
        if not self.quiet:
            columns = []
            for i, feature in enumerate(features, 1):
                columns.append(f"{StatusIcon.BULLET} {feature}")
            
            self.console.print(Columns(columns, equal=True, expand=True))
    
    def result(self, message: str, details: Optional[str] = None):
        """Display important results - always shown even in quiet mode."""
        self.console.print(f"{StatusIcon.SUCCESS} [bold green]{message}[/bold green]")
        if details and (self.verbose or not self.quiet):
            self._print_details(details, "green")
    
    def final_summary(self, title: str, items: Dict[str, Any], style: str = "green"):
        """Display final summary - always shown even in quiet mode."""
        content = []
        for key, value in items.items():
            content.append(f"[bold]{key}:[/bold] {value}")
        
        panel = Panel(
            "\n".join(content),
            title=f"[bold]{title}[/bold]",
            border_style=style,
            padding=(1, 2)
        )
        self.console.print()
        self.console.print(panel)
    
    def quiet_progress(self, message: str):
        """Show minimal progress in quiet mode - just the essential info."""
        if self.quiet:
            # In quiet mode, show very minimal progress indicators
            self.console.print(f"▶ {message}")
        else:
            # In normal mode, use the regular progress method
            self.progress(message)
    
    def _print_details(self, details: str, style: str, console: Optional[Console] = None):
        """Print details with proper indentation and styling."""
        target_console = console or self.console
        for line in details.split('\n'):
            if line.strip():
                target_console.print(f"  [dim]│[/dim] [{style}]{line}[/{style}]")
    
    def _get_status_icon(self, status: str) -> str:
        """Get appropriate icon for status."""
        status_lower = status.lower()
        if status_lower in ['success', 'passed', 'complete', 'done', 'ok']:
            return StatusIcon.SUCCESS
        elif status_lower in ['error', 'failed', 'fail']:
            return StatusIcon.ERROR
        elif status_lower in ['warning', 'warn']:
            return StatusIcon.WARNING
        elif status_lower in ['running', 'progress', 'processing']:
            return StatusIcon.PROGRESS
        elif status_lower in ['loading', 'analyzing']:
            return StatusIcon.ANALYZING
        elif status_lower in ['generating']:
            return StatusIcon.GENERATING
        elif status_lower in ['validating']:
            return StatusIcon.VALIDATING
        else:
            return StatusIcon.INFO

    def brand_header(self, subtitle: str = ""):
        """Display a concise header without marketing taglines."""
        if not self.quiet:
            # Create a sophisticated branded header
            title_text = Text()
            title_text.append("SlopTest", style="bold bright_blue")
            if subtitle:
                title_text.append(f" • {subtitle}", style="dim cyan")
            
            # Add version text
            version_text = "v1.0.0"
            
            header_panel = Panel(
                Align.center(title_text),
                subtitle=version_text,
                border_style="bright_blue",
                box=rich.box.DOUBLE,
                padding=(1, 2),
                title_align="center"
            )
            
            self.console.print()
            self.console.print(header_panel)
            
            # No tagline — keep output minimal and focused
    
    def execution_summary(self, mode: str, config: Dict[str, Any]):
        """Display a sophisticated execution summary."""
        if not self.quiet:
            # Create a modern execution card
            content = []
            content.append(f"[bold bright_cyan]Mode:[/bold bright_cyan] {mode.title()}")
            
            for key, value in config.items():
                if key == "Model":
                    content.append(f"[bold green]AI Model:[/bold green] {value}")
                elif key == "Directory":
                    content.append(f"[bold yellow]Target:[/bold yellow] {value}")
                elif key == "Batch Size":
                    content.append(f"[bold magenta]Batch Size:[/bold magenta] {value}")
                else:
                    content.append(f"[bold white]{key}:[/bold white] [dim]{value}[/dim]")
            
            panel = Panel(
                "\n".join(content),
                title="[bold bright_white]⚡ Execution Plan[/bold bright_white]",
                border_style="bright_green",
                box=rich.box.ROUNDED,
                padding=(1, 2),
                title_align="left"
            )
            self.console.print()
            self.console.print(panel)
    
    def sophisticated_progress(self, message: str, details: str = ""):
        """Display sophisticated progress with context."""
        if not self.quiet:
            progress_text = Text()
            progress_text.append("▶ ", style="bright_cyan bold")
            progress_text.append(message, style="white")
            if details:
                progress_text.append(f" • {details}", style="dim white")
            
            self.console.print(progress_text)
    
    def compact_progress(self, current: int, total: int, message: str = ""):
        """Display very compact progress for batch operations."""
        if not self.quiet:
            if total > 0:
                percentage = int(current / total * 100)
                progress_text = f"▶ {message} [{current}/{total}] {percentage}%"
                # Use carriage return to overwrite previous line for compact display
                self.console.print(f"\r{progress_text}", end="")
                if current == total:
                    self.console.print()  # Final newline when complete
    
    def test_plans_display(self, test_plans: List, project_root: Path = None):
        """Display test generation plans in a beautiful format."""
        if not test_plans or self.quiet:
            return
            
        # Calculate summary statistics
        total_elements = sum(len(plan.elements_to_test) for plan in test_plans)
        total_files = len(test_plans)
        files_with_existing_tests = len([p for p in test_plans if p.existing_test_files])
        
        # Display overview panel
        overview_content = []
        overview_content.append(f"[bold bright_cyan]Test Plans:[/bold bright_cyan] {total_files}")
        overview_content.append(f"[bold bright_yellow]Elements Needing Tests:[/bold bright_yellow] {total_elements}")
        overview_content.append(f"[bold bright_green]Files with Existing Tests:[/bold bright_green] {files_with_existing_tests}")
        if files_with_existing_tests > 0:
            overview_content.append(f"[bold bright_blue]New Test Files:[/bold bright_blue] {total_files - files_with_existing_tests}")
        
        overview_panel = Panel(
            "\n".join(overview_content),
            title="[bold bright_white]🎯 Test Generation Overview[/bold bright_white]",
            border_style="bright_blue",
            box=rich.box.ROUNDED,
            padding=(1, 2),
            title_align="left"
        )
        
        self.console.print()
        self.console.print(overview_panel)
        
        # Display detailed test plans table
        if total_elements > 0:
            table = Table(
                title="📋 Test Generation Plans",
                show_header=True, 
                header_style="bold magenta",
                border_style="cyan"
            )
            table.add_column("File", style="bright_cyan", min_width=25)
            table.add_column("Elements", style="bright_yellow", justify="center", width=8)
            table.add_column("Current Coverage", style="bright_green", justify="center", width=12)
            table.add_column("Estimated After", style="bright_blue", justify="center", width=12)
            table.add_column("Existing Tests", style="dim white", justify="center", width=8)
            
            for plan in test_plans:
                if plan.elements_to_test:  # Only show plans that need work
                    # Format file path
                    if project_root:
                        try:
                            relative_path = str(Path(plan.source_file).relative_to(project_root))
                        except ValueError:
                            relative_path = str(Path(plan.source_file).name)
                    else:
                        relative_path = str(Path(plan.source_file).name)
                    
                    # Format coverage info
                    current_coverage = f"{plan.coverage_before.line_coverage:.1f}%" if plan.coverage_before else "N/A"
                    estimated_coverage = f"{plan.estimated_coverage_after:.1f}%"
                    
                    # Existing tests count
                    existing_count = len(plan.existing_test_files)
                    existing_display = f"{existing_count}" if existing_count > 0 else "None"
                    
                    table.add_row(
                        relative_path,
                        str(len(plan.elements_to_test)),
                        current_coverage,
                        estimated_coverage,
                        existing_display
                    )
            
            self.console.print()
            self.console.print(table)
            
            # Show sample elements (for verbose mode or when not too many)
            if self.verbose or total_elements <= 20:
                self.console.print()
                for plan in test_plans[:5]:  # Show first 5 plans
                    if plan.elements_to_test:
                        file_name = Path(plan.source_file).name
                        elements_text = Text()
                        elements_text.append(f"📁 {file_name}: ", style="bold bright_cyan")
                        
                        # Show first few elements
                        element_names = []
                        for element in plan.elements_to_test[:3]:
                            element_names.append(f"{element.type}:{element.name}")
                        
                        elements_text.append(", ".join(element_names), style="dim white")
                        
                        if len(plan.elements_to_test) > 3:
                            elements_text.append(f" + {len(plan.elements_to_test) - 3} more...", style="dim yellow")
                        
                        self.console.print(elements_text)
                
                if len(test_plans) > 5:
                    remaining = len(test_plans) - 5
                    self.console.print(f"[dim]... and {remaining} more files[/dim]")

    def verbose_prompt_display(self, model_name: str, system_prompt: str, user_content: str, 
                               content_size: int = None, file_count: int = None):
        """Display LLM prompts in verbose mode with beautiful formatting."""
        if not self.verbose:
            return
            
        self.console.print()
        self.console.print(Rule(f"🤖 LLM Prompt to {model_name}", style="bright_magenta"))
        
        # Show overview
        overview_content = []
        overview_content.append(f"[bold bright_blue]Model:[/bold bright_blue] {model_name}")
        if content_size:
            overview_content.append(f"[bold bright_yellow]Content Size:[/bold bright_yellow] {content_size:,} characters")
        if file_count:
            overview_content.append(f"[bold bright_green]Files Processing:[/bold bright_green] {file_count}")
        
        overview_panel = Panel(
            "\n".join(overview_content),
            title="[bold bright_white]📋 Request Overview[/bold bright_white]",
            border_style="bright_blue",
            box=rich.box.ROUNDED,
            padding=(0, 1),
            title_align="left"
        )
        
        self.console.print(overview_panel)
        
        # Display system prompt with syntax highlighting
        self.console.print()
        self.console.print("[bold bright_cyan]📝 System Prompt[/bold bright_cyan]")
        
        # Truncate very long prompts for display
        display_system = system_prompt
        if len(system_prompt) > 2000:
            display_system = system_prompt[:2000] + f"\n\n[... truncated {len(system_prompt) - 2000:,} more characters ...]"
        
        system_panel = Panel(
            display_system,
            border_style="cyan",
            box=rich.box.MINIMAL,
            padding=(1, 1)
        )
        self.console.print(system_panel)
        
        # Display user content with syntax highlighting 
        self.console.print()
        self.console.print("[bold bright_yellow]💬 User Content[/bold bright_yellow]")
        
        # Truncate very long user content for display
        display_user = user_content
        if len(user_content) > 3000:
            # Show first part and structure info
            truncated_content = user_content[:1500]
            
            # Try to find structure markers to show
            structure_markers = []
            if "<directory_structure>" in user_content:
                structure_markers.append("📁 Directory Structure")
            if "<code_files>" in user_content:
                structure_markers.append("📄 Code Files")
            if "AVAILABLE_IMPORTS" in user_content:
                structure_markers.append("📦 Available Imports")
            if "CLASS_SIGNATURES" in user_content:
                structure_markers.append("🏗️ Class Signatures")
            
            display_user = truncated_content + f"\n\n[... content truncated ...]\n\n"
            if structure_markers:
                display_user += f"Content includes: {', '.join(structure_markers)}\n"
            display_user += f"Total size: {len(user_content):,} characters"
        
        user_panel = Panel(
            display_user,
            border_style="yellow",
            box=rich.box.MINIMAL,
            padding=(1, 1)
        )
        self.console.print(user_panel)
        
        self.console.print(Rule(style="dim"))

    def quality_analysis_display(self, quality_reports: Dict[str, Any], project_root: Path = None):
        """Display test quality analysis results in a beautiful format."""
        if not quality_reports or self.quiet:
            return
            
        self.console.print()
        self.console.print(Rule("🎯 Test Quality Analysis", style="bright_magenta"))
        
        # Create overall summary
        total_files = len(quality_reports)
        all_scores = []
        high_quality_files = 0
        medium_quality_files = 0
        low_quality_files = 0
        
        for report in quality_reports.values():
            if hasattr(report, 'quality_report') and report.quality_report:
                score = report.quality_report.overall_score
                all_scores.append(score)
                if score >= 85:
                    high_quality_files += 1
                elif score >= 70:
                    medium_quality_files += 1
                else:
                    low_quality_files += 1
        
        if all_scores:
            average_score = sum(all_scores) / len(all_scores)
            
            # Display overall summary
            summary_content = []
            summary_content.append(f"[bold bright_cyan]Files Analyzed:[/bold bright_cyan] {total_files}")
            summary_content.append(f"[bold bright_green]Average Quality Score:[/bold bright_green] {average_score:.1f}%")
            summary_content.append(f"[bold green]High Quality (85%+):[/bold green] {high_quality_files}")
            summary_content.append(f"[bold yellow]Medium Quality (70-84%):[/bold yellow] {medium_quality_files}")
            summary_content.append(f"[bold red]Needs Improvement (<70%):[/bold red] {low_quality_files}")
            
            # Color-code the border based on average quality
            if average_score >= 85:
                border_color = "bright_green"
            elif average_score >= 70:
                border_color = "yellow"
            else:
                border_color = "red"
            
            summary_panel = Panel(
                "\n".join(summary_content),
                title="[bold bright_white]📊 Quality Overview[/bold bright_white]",
                border_style=border_color,
                box=rich.box.ROUNDED,
                padding=(1, 2),
                title_align="left"
            )
            
            self.console.print(summary_panel)
            
            # Display detailed quality breakdown table
            if total_files > 0:
                table = Table(
                    title="📋 Detailed Quality Analysis",
                    show_header=True,
                    header_style="bold magenta",
                    border_style="cyan"
                )
                table.add_column("Test File", style="bright_cyan", min_width=30)
                table.add_column("Overall Score", style="bold", justify="center", width=12)
                table.add_column("Edge Cases", style="green", justify="center", width=10)
                table.add_column("Assertions", style="blue", justify="center", width=10)
                table.add_column("Maintainability", style="yellow", justify="center", width=13)
                table.add_column("Independence", style="magenta", justify="center", width=12)
                table.add_column("Top Issue", style="dim white", min_width=25)
                
                for source_file, report in quality_reports.items():
                    if hasattr(report, 'quality_report') and report.quality_report:
                        quality_report = report.quality_report
                        
                        # Format file path
                        if project_root:
                            try:
                                relative_path = str(Path(quality_report.test_file).relative_to(project_root))
                            except ValueError:
                                relative_path = str(Path(quality_report.test_file).name)
                        else:
                            relative_path = str(Path(quality_report.test_file).name)
                        
                        # Get overall score with color
                        overall_score = quality_report.overall_score
                        if overall_score >= 85:
                            score_display = f"[bold green]{overall_score:.1f}%[/bold green]"
                        elif overall_score >= 70:
                            score_display = f"[bold yellow]{overall_score:.1f}%[/bold yellow]"
                        else:
                            score_display = f"[bold red]{overall_score:.1f}%[/bold red]"
                        
                        # Get dimension scores
                        from smart_test_generator.models.data_models import QualityDimension
                        edge_case_score = quality_report.get_score(QualityDimension.EDGE_CASE_COVERAGE)
                        assertion_score = quality_report.get_score(QualityDimension.ASSERTION_STRENGTH)  
                        maintainability_score = quality_report.get_score(QualityDimension.MAINTAINABILITY)
                        independence_score = quality_report.get_score(QualityDimension.INDEPENDENCE)
                        
                        # Get top priority issue
                        top_issue = "None"
                        if quality_report.priority_fixes:
                            top_issue = quality_report.priority_fixes[0][:40] + "..." if len(quality_report.priority_fixes[0]) > 40 else quality_report.priority_fixes[0]
                        
                        table.add_row(
                            relative_path,
                            score_display,
                            f"{edge_case_score:.1f}%",
                            f"{assertion_score:.1f}%", 
                            f"{maintainability_score:.1f}%",
                            f"{independence_score:.1f}%",
                            top_issue
                        )
                
                self.console.print()
                self.console.print(table)
                
                # Show priority fixes and suggestions
                all_priority_fixes = []
                all_suggestions = []
                
                for report in quality_reports.values():
                    if hasattr(report, 'quality_report') and report.quality_report:
                        all_priority_fixes.extend(report.quality_report.priority_fixes)
                        all_suggestions.extend(report.quality_report.improvement_suggestions)
                
                # Display priority fixes if any
                if all_priority_fixes:
                    self.console.print()
                    fixes_content = []
                    unique_fixes = list(dict.fromkeys(all_priority_fixes))  # Remove duplicates
                    for i, fix in enumerate(unique_fixes[:5], 1):  # Show top 5
                        fixes_content.append(f"[bold red]{i}.[/bold red] {fix}")
                    
                    fixes_panel = Panel(
                        "\n".join(fixes_content),
                        title="[bold bright_red]🚨 Priority Fixes Needed[/bold bright_red]",
                        border_style="red",
                        box=rich.box.ROUNDED,
                        padding=(1, 2),
                        title_align="left"
                    )
                    self.console.print(fixes_panel)
                
                # Display improvement suggestions
                if all_suggestions and self.verbose:
                    self.console.print()
                    suggestions_content = []
                    unique_suggestions = list(dict.fromkeys(all_suggestions))  # Remove duplicates
                    for i, suggestion in enumerate(unique_suggestions[:3], 1):  # Show top 3
                        suggestions_content.append(f"[bold blue]{i}.[/bold blue] {suggestion}")
                    
                    suggestions_panel = Panel(
                        "\n".join(suggestions_content),
                        title="[bold bright_blue]💡 Improvement Suggestions[/bold bright_blue]",
                        border_style="blue",
                        box=rich.box.ROUNDED,
                        padding=(1, 2),
                        title_align="left"
                    )
                    self.console.print(suggestions_panel)
        
        self.console.print()
        self.console.print(Rule(style="dim"))

    def completion_celebration(self, title: str, stats: Dict[str, Any], duration: str = ""):
        """Display a celebratory completion summary."""
        # Always show completion, even in quiet mode
        
        # Create celebration header
        celebration_text = Text()
        celebration_text.append("🎉 ", style="bright_green")
        celebration_text.append(title, style="bold bright_green")
        celebration_text.append(" Complete!", style="bright_green")
        
        if duration:
            celebration_text.append(f" ({duration})", style="dim bright_green")
        
        self.console.print()
        self.console.print(celebration_text)
        
        # Create sophisticated results panel
        if stats:
            content = []
            for key, value in stats.items():
                if "improvement" in key.lower() or "coverage" in key.lower():
                    content.append(f"[bold bright_green]{key}:[/bold bright_green] [bright_green]{value}[/bright_green]")
                elif "generated" in key.lower() or "created" in key.lower():
                    content.append(f"[bold bright_cyan]{key}:[/bold bright_cyan] [bright_cyan]{value}[/bright_cyan]")
                elif "files" in key.lower() or "processed" in key.lower():
                    content.append(f"[bold bright_yellow]{key}:[/bold bright_yellow] [bright_yellow]{value}[/bright_yellow]")
                else:
                    content.append(f"[bold white]{key}:[/bold white] {value}")
            
            results_panel = Panel(
                "\n".join(content),
                title="[bold bright_white]📊 Results Summary[/bold bright_white]",
                border_style="bright_green",
                box=rich.box.HEAVY,
                padding=(1, 2),
                title_align="left"
            )
            
            self.console.print(results_panel)
            
            # Add a success indicator
            if not self.quiet:
                success_text = Text()
                success_text.append("✨ ", style="bright_yellow")
                success_text.append("Ready for testing! ", style="bold white")
                success_text.append("Your generated tests are waiting in the tests/ directory.", style="dim white")
                self.console.print()
                self.console.print(Align.center(success_text))
                self.console.print() 


class ProgressTracker:
    """Enhanced progress tracker with rich progress bars."""
    
    def __init__(self, feedback: UserFeedback):
        self.feedback = feedback
        self.total_steps = 0
        self.current_step = 0
        self.step_messages = []
        self._progress = None
        self._task_id = None
        self._use_rich_progress = True
    
    def set_total_steps(self, total: int, description: str = "Processing"):
        """Set the total number of steps and start progress tracking."""
        self.total_steps = total
        self.current_step = 0
        
        # Clean up any existing progress first
        if self._progress:
            try:
                self._progress.stop()
            except:
                pass
            self._progress = None
            self._task_id = None
        
        # Initialize rich Progress; on failure, fall back to simple indicators
        try:
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                TimeElapsedColumn(),
                console=self.feedback.console,
                transient=False,
            )
            progress.start()
            task_id = progress.add_task(description, total=total, completed=0)
            self._progress = progress
            self._task_id = task_id
            self._use_rich_progress = True
        except Exception:
            self._use_rich_progress = False
            self._progress = None
            self._task_id = None
            self.feedback.info(f"Starting: {description}")
    
    def step(self, message: str):
        """Complete a step and update progress."""
        self.current_step += 1
        self.step_messages.append(message)
        
        if self._use_rich_progress and self._progress and self._task_id is not None:
            try:
                description = f"Step {self.current_step}/{self.total_steps}: {message}"
                self._progress.update(self._task_id, advance=1, description=description)
                return
            except Exception:
                # Fall back if update fails
                self._use_rich_progress = False
        
        # Fallback simple indicator
        self.feedback.step_indicator(self.current_step, self.total_steps, message)
    
    def complete(self, success_message: str):
        """Mark all steps as complete."""
        if self._progress and self._use_rich_progress:
            try:
                remaining = max(0, self.total_steps - self.current_step)
                if remaining > 0 and self._task_id is not None:
                    self._progress.update(self._task_id, advance=remaining)
            except Exception:
                pass
            try:
                self._progress.stop()
            except Exception:
                pass
            finally:
                self._progress = None
                self._task_id = None
                
        self.feedback.success(success_message)
    
    def error(self, error_message: str):
        """Mark progress as failed."""
        if self._progress and self._use_rich_progress:
            try:
                self._progress.stop()
            except Exception:
                pass
            finally:
                self._progress = None
                self._task_id = None
        self.feedback.error(error_message) 