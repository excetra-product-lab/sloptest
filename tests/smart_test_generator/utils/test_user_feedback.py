import pytest
from unittest.mock import Mock, patch, MagicMock, call
from pathlib import Path
from contextlib import contextmanager
from typing import Dict, List, Tuple, Any

from smart_test_generator.utils.user_feedback import (
    UserFeedback,
    ProgressTracker,
    MessageType,
    StatusIcon
)


class TestUserFeedback:
    """Test UserFeedback class."""
    
    def test_init_default_values(self):
        """Test UserFeedback initialization with default values."""
        feedback = UserFeedback()
        
        assert feedback.verbose is False
        assert feedback.console is not None
        assert feedback.error_console is not None
        assert feedback._current_progress is None
        assert feedback._status_tracking == {}
    
    def test_init_with_verbose_true(self):
        """Test UserFeedback initialization with verbose=True."""
        feedback = UserFeedback(verbose=True)
        
        assert feedback.verbose is True
        assert feedback.console is not None
        assert feedback.error_console is not None
    
    @patch('smart_test_generator.utils.user_feedback.Console')
    def test_init_console_configuration(self, mock_console):
        """Test that consoles are configured correctly."""
        UserFeedback()
        
        # Check that Console was called twice (once for regular, once for error)
        assert mock_console.call_count == 2
        
        # Check the calls
        calls = mock_console.call_args_list
        assert calls[0] == call(stderr=False, force_terminal=True)
        assert calls[1] == call(stderr=True, force_terminal=True)
    
    def test_success_basic_message(self):
        """Test success method with basic message."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.success("Test completed")
        
        feedback.console.print.assert_called_once_with(f"{StatusIcon.SUCCESS} Test completed")
    
    def test_success_with_details_verbose_false(self):
        """Test success method with details when verbose is False."""
        feedback = UserFeedback(verbose=False)
        feedback.console = Mock()
        feedback._print_details = Mock()
        
        feedback.success("Test completed", details="Additional info")
        
        feedback.console.print.assert_called_once_with(f"{StatusIcon.SUCCESS} Test completed")
        feedback._print_details.assert_not_called()
    
    def test_success_with_details_verbose_true(self):
        """Test success method with details when verbose is True."""
        feedback = UserFeedback(verbose=True)
        feedback.console = Mock()
        feedback._print_details = Mock()
        
        feedback.success("Test completed", details="Additional info")
        
        feedback.console.print.assert_called_once_with(f"{StatusIcon.SUCCESS} Test completed")
        feedback._print_details.assert_called_once_with("Additional info", "green")
    
    def test_error_basic_message(self):
        """Test error method with basic message."""
        feedback = UserFeedback()
        feedback.error_console = Mock()
        
        feedback.error("Something went wrong")
        
        feedback.error_console.print.assert_called_once_with(
            f"{StatusIcon.ERROR} [bold red]Error:[/bold red] Something went wrong"
        )
    
    def test_error_with_suggestion(self):
        """Test error method with suggestion."""
        feedback = UserFeedback()
        feedback.error_console = Mock()
        
        feedback.error("Something went wrong", suggestion="Try this fix")
        
        expected_calls = [
            call(f"{StatusIcon.ERROR} [bold red]Error:[/bold red] Something went wrong"),
            call("  [yellow]💡 Suggestion:[/yellow] Try this fix")
        ]
        feedback.error_console.print.assert_has_calls(expected_calls)
    
    def test_error_with_details_verbose_true(self):
        """Test error method with details when verbose is True."""
        feedback = UserFeedback(verbose=True)
        feedback.error_console = Mock()
        feedback._print_details = Mock()
        
        feedback.error("Something went wrong", details="Stack trace here")
        
        feedback.error_console.print.assert_called_once_with(
            f"{StatusIcon.ERROR} [bold red]Error:[/bold red] Something went wrong"
        )
        feedback._print_details.assert_called_once_with("Stack trace here", "red", console=feedback.error_console)
    
    def test_error_with_suggestion_and_details(self):
        """Test error method with both suggestion and details."""
        feedback = UserFeedback(verbose=True)
        feedback.error_console = Mock()
        feedback._print_details = Mock()
        
        feedback.error("Something went wrong", suggestion="Try this fix", details="Stack trace here")
        
        expected_calls = [
            call(f"{StatusIcon.ERROR} [bold red]Error:[/bold red] Something went wrong"),
            call("  [yellow]💡 Suggestion:[/yellow] Try this fix")
        ]
        feedback.error_console.print.assert_has_calls(expected_calls)
        feedback._print_details.assert_called_once_with("Stack trace here", "red", console=feedback.error_console)
    
    def test_warning_basic_message(self):
        """Test warning method with basic message."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.warning("This might be an issue")
        
        feedback.console.print.assert_called_once_with(
            f"{StatusIcon.WARNING} [bold yellow]Warning:[/bold yellow] This might be an issue"
        )
    
    def test_warning_with_suggestion(self):
        """Test warning method with suggestion."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.warning("This might be an issue", suggestion="Consider doing this")
        
        expected_calls = [
            call(f"{StatusIcon.WARNING} [bold yellow]Warning:[/bold yellow] This might be an issue"),
            call("  [yellow]💡 Consider doing this[/yellow]")
        ]
        feedback.console.print.assert_has_calls(expected_calls)
    
    def test_info_basic_message(self):
        """Test info method with basic message."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.info("Here's some information")
        
        feedback.console.print.assert_called_once_with(f"{StatusIcon.INFO} Here's some information")
    
    def test_info_with_details_verbose_true(self):
        """Test info method with details when verbose is True."""
        feedback = UserFeedback(verbose=True)
        feedback.console = Mock()
        feedback._print_details = Mock()
        
        feedback.info("Here's some information", details="More details")
        
        feedback.console.print.assert_called_once_with(f"{StatusIcon.INFO} Here's some information")
        feedback._print_details.assert_called_once_with("More details", "blue")
    
    def test_debug_verbose_false(self):
        """Test debug method when verbose is False."""
        feedback = UserFeedback(verbose=False)
        feedback.console = Mock()
        
        feedback.debug("Debug information")
        
        feedback.console.print.assert_not_called()
    
    def test_debug_verbose_true(self):
        """Test debug method when verbose is True."""
        feedback = UserFeedback(verbose=True)
        feedback.console = Mock()
        
        feedback.debug("Debug information")
        
        feedback.console.print.assert_called_once_with(f"{StatusIcon.DEBUG} [dim]Debug information[/dim]")
    
    def test_debug_with_details_verbose_true(self):
        """Test debug method with details when verbose is True."""
        feedback = UserFeedback(verbose=True)
        feedback.console = Mock()
        feedback._print_details = Mock()
        
        feedback.debug("Debug information", details="Debug details")
        
        feedback.console.print.assert_called_once_with(f"{StatusIcon.DEBUG} [dim]Debug information[/dim]")
        feedback._print_details.assert_called_once_with("Debug details", "dim")
    
    def test_progress_message(self):
        """Test progress method displays message with progress icon."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.progress("Processing files")
        
        feedback.console.print.assert_called_once_with(f"{StatusIcon.PROGRESS} Processing files")
    
    @patch('smart_test_generator.utils.user_feedback.Panel')
    @patch('smart_test_generator.utils.user_feedback.Align')
    @patch('smart_test_generator.utils.user_feedback.Text')
    def test_section_header(self, mock_text, mock_align, mock_panel):
        """Test section_header creates and prints a panel."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        mock_text_instance = Mock()
        mock_text.return_value = mock_text_instance
        mock_align_instance = Mock()
        mock_align.center.return_value = mock_align_instance
        mock_panel_instance = Mock()
        mock_panel.return_value = mock_panel_instance
        
        feedback.section_header("Test Section")
        
        mock_text.assert_called_once_with("Test Section", style="bold white")
        mock_align.center.assert_called_once_with(mock_text_instance)
        mock_panel.assert_called_once_with(
            mock_align_instance,
            border_style="bright_blue",
            padding=(0, 1),
            title_align="center"
        )
        
        expected_calls = [call(), call(mock_panel_instance)]
        feedback.console.print.assert_has_calls(expected_calls)
    
    def test_subsection_header(self):
        """Test subsection_header displays formatted title."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.subsection_header("Sub Section")
        
        feedback.console.print.assert_called_once_with("\n[bold bright_cyan]▊ Sub Section[/bold bright_cyan]")
    
    @patch('smart_test_generator.utils.user_feedback.Table')
    def test_status_table(self, mock_table):
        """Test status_table creates and displays a table."""
        feedback = UserFeedback()
        feedback.console = Mock()
        feedback._get_status_icon = Mock(side_effect=lambda x: f"icon_{x}")
        
        mock_table_instance = Mock()
        mock_table.return_value = mock_table_instance
        
        items = [
            ("success", "Component A", "Working fine"),
            ("error", "Component B", "Has issues")
        ]
        
        feedback.status_table("System Status", items)
        
        mock_table.assert_called_once_with(title="System Status", show_header=True, header_style="bold magenta")
        mock_table_instance.add_column.assert_any_call("Status", style="bold", width=8, justify="center")
        mock_table_instance.add_column.assert_any_call("Component", style="cyan", min_width=20)
        mock_table_instance.add_column.assert_any_call("Description", style="white")
        
        mock_table_instance.add_row.assert_any_call("icon_success", "Component A", "Working fine")
        mock_table_instance.add_row.assert_any_call("icon_error", "Component B", "Has issues")
        
        feedback.console.print.assert_called_once_with(mock_table_instance)
    
    @patch('smart_test_generator.utils.user_feedback.Panel')
    def test_summary_panel(self, mock_panel):
        """Test summary_panel creates and displays a panel with items."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        mock_panel_instance = Mock()
        mock_panel.return_value = mock_panel_instance
        
        items = {"Tests": 42, "Coverage": "85%"}
        
        feedback.summary_panel("Summary", items, "blue")
        
        expected_content = "[bold]Tests:[/bold] 42\n[bold]Coverage:[/bold] 85%"
        mock_panel.assert_called_once_with(
            expected_content,
            title="Summary",
            border_style="blue",
            padding=(1, 2)
        )
        feedback.console.print.assert_called_once_with(mock_panel_instance)
    
    @patch('smart_test_generator.utils.user_feedback.Tree')
    def test_file_tree_basic(self, mock_tree):
        """Test file_tree creates and displays a tree structure."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        mock_tree_instance = Mock()
        mock_tree.return_value = mock_tree_instance
        
        files = [Path("test.py"), Path("utils.py")]
        
        feedback.file_tree("Files", files)
        
        mock_tree.assert_called_once_with("[bold blue]Files[/bold blue]")
        feedback.console.print.assert_called_once_with(mock_tree_instance)
    
    @patch('smart_test_generator.utils.user_feedback.Tree')
    def test_file_tree_with_base_path(self, mock_tree):
        """Test file_tree with base_path parameter."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        mock_tree_instance = Mock()
        mock_tree.return_value = mock_tree_instance
        
        base_path = Path("/project")
        files = [Path("/project/src/test.py"), Path("/project/utils.py")]
        
        feedback.file_tree("Files", files, base_path)
        
        mock_tree.assert_called_once_with("[bold blue]Files[/bold blue]")
        feedback.console.print.assert_called_once_with(mock_tree_instance)
    
    def test_status_spinner(self):
        """Test status_spinner context manager."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        # Mock the status context manager
        mock_status = Mock()
        feedback.console.status.return_value = mock_status
        mock_status.__enter__ = Mock(return_value=mock_status)
        mock_status.__exit__ = Mock(return_value=None)
        
        with feedback.status_spinner("Loading", "dots") as status:
            assert status == mock_status
        
        feedback.console.status.assert_called_once_with(f"{StatusIcon.LOADING} Loading", spinner="dots")
        mock_status.__enter__.assert_called_once()
        mock_status.__exit__.assert_called_once()
    
    @patch('smart_test_generator.utils.user_feedback.Progress')
    def test_progress_context(self, mock_progress_class):
        """Test progress_context creates and yields Progress instance."""
        feedback = UserFeedback()
        
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress
        mock_progress.__enter__ = Mock(return_value=mock_progress)
        mock_progress.__exit__ = Mock(return_value=None)
        
        with feedback.progress_context("Processing") as progress:
            assert progress == mock_progress
        
        mock_progress_class.assert_called_once()
        mock_progress.__enter__.assert_called_once()
        mock_progress.__exit__.assert_called_once()
    
    def test_progress_bar_zero_total(self):
        """Test progress_bar handles zero total gracefully."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        # Should return early without doing anything
        feedback.progress_bar(5, 0, "Processing")
        
        feedback.console.status.assert_not_called()
    
    @patch('time.sleep')
    def test_progress_bar_normal_operation(self, mock_sleep):
        """Test progress_bar displays progress correctly."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        mock_status = Mock()
        feedback.console.status.return_value = mock_status
        mock_status.__enter__ = Mock(return_value=mock_status)
        mock_status.__exit__ = Mock(return_value=None)
        
        feedback.progress_bar(3, 10, "Processing")
        
        expected_message = "Processing [3/10] 30.0%"
        feedback.console.status.assert_called_once_with(expected_message)
        mock_sleep.assert_called_once_with(0.1)
    
    @patch('smart_test_generator.utils.user_feedback.Table')
    def test_operation_status(self, mock_table):
        """Test operation_status displays operations table."""
        feedback = UserFeedback()
        feedback.console = Mock()
        feedback._get_status_icon = Mock(side_effect=lambda x: f"icon_{x}")
        
        mock_table_instance = Mock()
        mock_table.return_value = mock_table_instance
        
        operations = {"Analysis": "complete", "Testing": "running"}
        
        feedback.operation_status(operations)
        
        mock_table.assert_called_once_with(show_header=True, header_style="bold magenta")
        mock_table_instance.add_row.assert_any_call("Analysis", "icon_complete", "complete")
        mock_table_instance.add_row.assert_any_call("Testing", "icon_running", "running")
        feedback.console.print.assert_called_once_with(mock_table_instance)
    
    @patch('smart_test_generator.utils.user_feedback.Confirm')
    def test_confirm_default_false(self, mock_confirm):
        """Test confirm method with default False."""
        feedback = UserFeedback()
        mock_confirm.ask.return_value = True
        
        result = feedback.confirm("Continue?", default=False)
        
        assert result is True
        mock_confirm.ask.assert_called_once_with("Continue?", default=False, console=feedback.console)
    
    @patch('smart_test_generator.utils.user_feedback.Confirm')
    def test_confirm_default_true(self, mock_confirm):
        """Test confirm method with default True."""
        feedback = UserFeedback()
        mock_confirm.ask.return_value = False
        
        result = feedback.confirm("Continue?", default=True)
        
        assert result is False
        mock_confirm.ask.assert_called_once_with("Continue?", default=True, console=feedback.console)
    
    def test_summary_alias_for_summary_panel(self):
        """Test summary method is an alias for summary_panel."""
        feedback = UserFeedback()
        feedback.summary_panel = Mock()
        
        items = {"key": "value"}
        feedback.summary("Title", items, "red")
        
        feedback.summary_panel.assert_called_once_with("Title", items, "red")
    
    def test_step_indicator(self):
        """Test step_indicator displays step progress."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.step_indicator(2, 5, "Running tests")
        
        expected_message = f"{StatusIcon.ARROW_RIGHT} [bold cyan]Step 2/5[/bold cyan]: Running tests"
        feedback.console.print.assert_called_once_with(expected_message)
    
    def test_divider_with_text(self):
        """Test divider method with text."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.divider("Section")
        
        expected_message = "\n[dim]──────────────────── Section ────────────────────[/dim]"
        feedback.console.print.assert_called_once_with(expected_message)
    
    def test_divider_without_text(self):
        """Test divider method without text."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        feedback.divider()
        
        expected_message = "[dim]────────────────────────────────────────────────────────[/dim]"
        feedback.console.print.assert_called_once_with(expected_message)
    
    @patch('smart_test_generator.utils.user_feedback.Columns')
    def test_feature_showcase(self, mock_columns):
        """Test feature_showcase displays features in columns."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        mock_columns_instance = Mock()
        mock_columns.return_value = mock_columns_instance
        
        features = ["Feature A", "Feature B", "Feature C"]
        
        feedback.feature_showcase(features)
        
        expected_columns = [
            f"{StatusIcon.BULLET} Feature A",
            f"{StatusIcon.BULLET} Feature B",
            f"{StatusIcon.BULLET} Feature C"
        ]
        mock_columns.assert_called_once_with(expected_columns, equal=True, expand=True)
        feedback.console.print.assert_called_once_with(mock_columns_instance)
    
    def test_print_details_default_console(self):
        """Test _print_details with default console."""
        feedback = UserFeedback()
        feedback.console = Mock()
        
        details = "Line 1\nLine 2\n\nLine 4"
        
        feedback._print_details(details, "green")
        
        expected_calls = [
            call("  [dim]│[/dim] [green]Line 1[/green]"),
            call("  [dim]│[/dim] [green]Line 2[/green]"),
            call("  [dim]│[/dim] [green]Line 4[/green]")
        ]
        feedback.console.print.assert_has_calls(expected_calls)
    
    def test_print_details_custom_console(self):
        """Test _print_details with custom console."""
        feedback = UserFeedback()
        custom_console = Mock()
        
        details = "Error details"
        
        feedback._print_details(details, "red", console=custom_console)
        
        custom_console.print.assert_called_once_with("  [dim]│[/dim] [red]Error details[/red]")
    
    def test_get_status_icon_success_statuses(self):
        """Test _get_status_icon returns success icon for success statuses."""
        feedback = UserFeedback()
        
        success_statuses = ['success', 'passed', 'complete', 'done', 'ok', 'SUCCESS', 'PASSED']
        
        for status in success_statuses:
            result = feedback._get_status_icon(status)
            assert result == StatusIcon.SUCCESS
    
    def test_get_status_icon_error_statuses(self):
        """Test _get_status_icon returns error icon for error statuses."""
        feedback = UserFeedback()
        
        error_statuses = ['error', 'failed', 'fail', 'ERROR', 'FAILED']
        
        for status in error_statuses:
            result = feedback._get_status_icon(status)
            assert result == StatusIcon.ERROR
    
    def test_get_status_icon_warning_statuses(self):
        """Test _get_status_icon returns warning icon for warning statuses."""
        feedback = UserFeedback()
        
        warning_statuses = ['warning', 'warn', 'WARNING']
        
        for status in warning_statuses:
            result = feedback._get_status_icon(status)
            assert result == StatusIcon.WARNING
    
    def test_get_status_icon_progress_statuses(self):
        """Test _get_status_icon returns progress icon for progress statuses."""
        feedback = UserFeedback()
        
        progress_statuses = ['running', 'progress', 'processing', 'RUNNING']
        
        for status in progress_statuses:
            result = feedback._get_status_icon(status)
            assert result == StatusIcon.PROGRESS
    
    def test_get_status_icon_analyzing_statuses(self):
        """Test _get_status_icon returns analyzing icon for analyzing statuses."""
        feedback = UserFeedback()
        
        analyzing_statuses = ['loading', 'analyzing', 'LOADING']
        
        for status in analyzing_statuses:
            result = feedback._get_status_icon(status)
            assert result == StatusIcon.ANALYZING
    
    def test_get_status_icon_generating_status(self):
        """Test _get_status_icon returns generating icon for generating status."""
        feedback = UserFeedback()
        
        result = feedback._get_status_icon('generating')
        assert result == StatusIcon.GENERATING
    
    def test_get_status_icon_validating_status(self):
        """Test _get_status_icon returns validating icon for validating status."""
        feedback = UserFeedback()
        
        result = feedback._get_status_icon('validating')
        assert result == StatusIcon.VALIDATING
    
    def test_get_status_icon_unknown_status(self):
        """Test _get_status_icon returns info icon for unknown status."""
        feedback = UserFeedback()
        
        result = feedback._get_status_icon('unknown_status')
        assert result == StatusIcon.INFO


class TestProgressTracker:
    """Test ProgressTracker class."""
    
    def test_init_with_feedback(self):
        """Test ProgressTracker initialization with feedback."""
        mock_feedback = Mock()
        tracker = ProgressTracker(mock_feedback)
        
        assert tracker.feedback == mock_feedback
        assert tracker.total_steps == 0
        assert tracker.current_step == 0
        assert tracker.step_messages == []
        assert tracker._progress is None
        assert tracker._task_id is None
        assert tracker._use_rich_progress is True
    
    @patch('smart_test_generator.utils.user_feedback.Progress')
    def test_set_total_steps_success(self, mock_progress_class):
        """Test set_total_steps successfully initializes progress."""
        mock_feedback = Mock()
        mock_progress = Mock()
        mock_progress_class.return_value = mock_progress
        mock_progress.add_task.return_value = "task_123"
        
        tracker = ProgressTracker(mock_feedback)
        tracker.set_total_steps(10, "Testing")
        
        assert tracker.total_steps == 10
        assert tracker.current_step == 0
        mock_progress.start.assert_called_once()
        mock_progress.add_task.assert_called_once_with("Testing", total=10, completed=0)
        assert tracker._task_id == "task_123"
        assert tracker._use_rich_progress is True
    
    @patch('smart_test_generator.utils.user_feedback.Progress')
    def test_set_total_steps_with_existing_progress(self, mock_progress_class):
        """Test set_total_steps cleans up existing progress first."""
        mock_feedback = Mock()
        old_progress = Mock()
        new_progress = Mock()
        mock_progress_class.return_value = new_progress
        new_progress.add_task.return_value = "new_task"
        
        tracker = ProgressTracker(mock_feedback)
        tracker._progress = old_progress
        
        tracker.set_total_steps(5, "New Task")
        
        old_progress.stop.assert_called_once()
        new_progress.start.assert_called_once()
        assert tracker._progress == new_progress
    
    @patch('smart_test_generator.utils.user_feedback.Progress')
    def test_set_total_steps_progress_fails(self, mock_progress_class):
        """Test set_total_steps falls back when Progress initialization fails."""
        mock_feedback = Mock()
        mock_progress_class.side_effect = Exception("Progress failed")
        
        tracker = ProgressTracker(mock_feedback)
        tracker.set_total_steps(10, "Testing")
        
        assert tracker.total_steps == 10
        assert tracker._use_rich_progress is False
        assert tracker._progress is None
        mock_feedback.info.assert_called_once_with("Starting: Testing")
    
    def test_step_with_rich_progress(self):
        """Test step method with rich progress enabled."""
        mock_feedback = Mock()
        mock_progress = Mock()
        
        tracker = ProgressTracker(mock_feedback)
        tracker._progress = mock_progress
        tracker._task_id = "task_123"
        tracker._use_rich_progress = True
        tracker.total_steps = 5
        
        tracker.step("Running test 1")
        
        assert tracker.current_step == 1
        assert "Running test 1" in tracker.step_messages
        mock_progress.update.assert_called_once_with(
            "task_123",
            advance=1,
            description="Step 1/5: Running test 1"
        )
    
    def test_step_with_rich_progress_update_fails(self):
        """Test step method falls back when progress update fails."""
        mock_feedback = Mock()
        mock_progress = Mock()
        mock_progress.update.side_effect = Exception("Update failed")
        
        tracker = ProgressTracker(mock_feedback)
        tracker._progress = mock_progress
        tracker._task_id = "task_123"
        tracker._use_rich_progress = True
        tracker.total_steps = 5
        
        tracker.step("Running test 1")
        
        assert tracker.current_step == 1
        assert tracker._use_rich_progress is False
        mock_feedback.step_indicator.assert_called_once_with(1, 5, "Running test 1")
    
    def test_step_without_rich_progress(self):
        """Test step method without rich progress."""
        mock_feedback = Mock()
        
        tracker = ProgressTracker(mock_feedback)
        tracker._use_rich_progress = False
        tracker.total_steps = 3
        
        tracker.step("Running test 1")
        
        assert tracker.current_step == 1
        mock_feedback.step_indicator.assert_called_once_with(1, 3, "Running test 1")
    
    def test_complete_with_rich_progress(self):
        """Test complete method with rich progress enabled."""
        mock_feedback = Mock()
        mock_progress = Mock()
        
        tracker = ProgressTracker(mock_feedback)
        tracker._progress = mock_progress
        tracker._task_id = "task_123"
        tracker._use_rich_progress = True
        tracker.total_steps = 5
        tracker.current_step = 5
        
        tracker.complete("All tests passed")
        
        mock_progress.stop.assert_called_once()
        mock_feedback.success.assert_called_once_with("All tests passed")
        assert tracker._progress is None
        assert tracker._task_id is None
    
    def test_complete_with_remaining_steps(self):
        """Test complete method advances remaining steps before stopping."""
        mock_feedback = Mock()
        mock_progress = Mock()
        
        tracker = ProgressTracker(mock_feedback)
        tracker._progress = mock_progress
        tracker._task_id = "task_123"
        tracker._use_rich_progress = True
        tracker.total_steps = 5
        tracker.current_step = 3  # 2 steps remaining
        
        tracker.complete("All tests passed")
        
        mock_progress.update.assert_called_once_with("task_123", advance=2)
        mock_progress.stop.assert_called_once()
        mock_feedback.success.assert_called_once_with("All tests passed")
    
    def test_complete_progress_stop_fails(self):
        """Test complete method handles progress stop failure gracefully."""
        mock_feedback = Mock()
        mock_progress = Mock()
        mock_progress.stop.side_effect = Exception("Stop failed")
        
        tracker = ProgressTracker(mock_feedback)
        tracker._progress = mock_progress
        tracker._use_rich_progress = True
        tracker.current_step = 5
        tracker.total_steps = 5
        
        tracker.complete("All tests passed")
        
        mock_progress.stop.assert_called_once()
        mock_feedback.success.assert_called_once_with("All tests passed")
        assert tracker._progress is None
    
    def test_complete_without_rich_progress(self):
        """Test complete method without rich progress."""
        mock_feedback = Mock()
        
        tracker = ProgressTracker(mock_feedback)
        tracker._use_rich_progress = False
        
        tracker.complete("All tests passed")
        
        mock_feedback.success.assert_called_once_with("All tests passed")
    
    def test_error_with_rich_progress(self):
        """Test error method with rich progress enabled."""
        mock_feedback = Mock()
        mock_progress = Mock()
        
        tracker = ProgressTracker(mock_feedback)
        tracker._progress = mock_progress
        tracker._use_rich_progress = True
        
        tracker.error("Test failed")
        
        mock_progress.stop.assert_called_once()
        mock_feedback.error.assert_called_once_with("Test failed")
        assert tracker._progress is None
        assert tracker._task_id is None
    
    def test_error_progress_stop_fails(self):
        """Test error method handles progress stop failure gracefully."""
        mock_feedback = Mock()
        mock_progress = Mock()
        mock_progress.stop.side_effect = Exception("Stop failed")
        
        tracker = ProgressTracker(mock_feedback)
        tracker._progress = mock_progress
        tracker._use_rich_progress = True
        
        tracker.error("Test failed")
        
        mock_progress.stop.assert_called_once()
        mock_feedback.error.assert_called_once_with("Test failed")
        assert tracker._progress is None
    
    def test_error_without_rich_progress(self):
        """Test error method without rich progress."""
        mock_feedback = Mock()
        
        tracker = ProgressTracker(mock_feedback)
        tracker._use_rich_progress = False
        
        tracker.error("Test failed")
        
        mock_feedback.error.assert_called_once_with("Test failed")
