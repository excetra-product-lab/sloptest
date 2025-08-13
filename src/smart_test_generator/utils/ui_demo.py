"""Demo utilities to showcase the world-class CLI UI."""

import time
from pathlib import Path

from .user_feedback import UserFeedback, ProgressTracker


def demo_world_class_ui() -> None:
    """Showcase the CLI features for demos and manual testing.

    This function intentionally focuses on calling the UI API to demonstrate
    capabilities; business logic is mocked.
    """
    feedback = UserFeedback(verbose=True)

    feedback.section_header("CLI Interface Showcase")

    # Feature showcase
    features = [
        "Professional status indicators",
        "Beautiful progress tracking",
        "Rich table displays",
        "File tree visualization",
        "Elegant panels and summaries",
    ]
    feedback.feature_showcase(features)
    feedback.divider("Status Table Demo")

    # Status messages
    feedback.subsection_header("Status Messages")
    feedback.success("This is a success message")
    feedback.info("This is an informational message")
    feedback.warning("This is a warning message", "Consider taking this action")
    feedback.debug("This is a debug message (only shown in verbose mode)")

    # Tables
    validation_items = [
        ("success", "System Check", "All requirements met"),
        ("success", "Dependencies", "All packages available"),
        ("warning", "Configuration", "Using default settings"),
        ("running", "File Analysis", "Scanning project files"),
        ("pending", "Test Generation", "Waiting for analysis"),
    ]
    feedback.status_table("Project Status", validation_items)
    feedback.divider("Configuration Panel Demo")

    # Summary panel (configuration)
    config_info = {
        "Project Root": "/path/to/project",
        "Test Directory": "tests/",
        "Coverage Threshold": "80%",
        "Model": "claude-sonnet-4",
        "Batch Size": "10 files",
        "Force Regeneration": "No",
    }
    feedback.summary_panel("Current Configuration", config_info, "blue")

    # Progress tracking
    feedback.divider("Progress Tracking Demo")
    tracker = ProgressTracker(feedback)
    tracker.set_total_steps(6, "Setup Progress")
    for step in [
        "Initializing system",
        "Scanning Python files",
        "Analyzing code structure",
        "Calculating coverage",
        "Planning test generation",
        "Finalizing setup",
    ]:
        tracker.step(step)
        time.sleep(0.5)
    tracker.complete("Setup completed successfully!")

    # File tree demo
    feedback.divider("File Tree Demo")
    files = [
        Path("src/smart_test_generator/__init__.py"),
        Path("src/smart_test_generator/cli.py"),
        Path("src/smart_test_generator/config.py"),
        Path("src/smart_test_generator/core/application.py"),
        Path("src/smart_test_generator/core/llm_factory.py"),
        Path("src/smart_test_generator/utils/user_feedback.py"),
        Path("src/smart_test_generator/services/analysis_service.py"),
        Path("tests/test_cli.py"),
        Path("tests/test_config.py"),
    ]
    feedback.file_tree("Project Structure", files, base_path=Path("src"))

    # Spinner demo
    feedback.divider("Spinner Demo")
    # Be tolerant to mocks that don't implement context manager protocol
    _ctx1 = feedback.status_spinner("Processing large dataset", spinner_style="dots12")
    if hasattr(_ctx1, "__enter__") and hasattr(_ctx1, "__exit__"):
        with _ctx1:
            time.sleep(2)
    else:
        time.sleep(2)
    feedback.success("Processing completed!")

    _ctx2 = feedback.status_spinner("Generating AI responses", spinner_style="arc")
    if hasattr(_ctx2, "__enter__") and hasattr(_ctx2, "__exit__"):
        with _ctx2:
            time.sleep(1.5)
    else:
        time.sleep(1.5)
    feedback.success("Generation completed!")

    # Operation status
    feedback.divider("Operation Status Demo")
    operations = {
        "File Discovery": "completed",
        "Code Analysis": "completed",
        "Coverage Calculation": "running",
        "Test Planning": "pending",
        "Quality Assessment": "pending",
    }
    feedback.operation_status(operations)

    # Final summary
    feedback.divider("Final Summary")
    final = {
        "Demo Mode": "Interactive Showcase",
        "Features Demonstrated": "8 categories",
        "Status": "All features working",
        "Performance": "Excellent",
    }
    feedback.summary_panel("Demo Summary", final, "green")
    feedback.success("CLI interface showcase completed!")
    feedback.info("Ready for production use with beautiful, professional interface")

