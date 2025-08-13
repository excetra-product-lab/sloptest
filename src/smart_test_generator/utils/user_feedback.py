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
            title_text.append("Smart Test Generator", style="bold bright_blue")
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