#!/usr/bin/env python3
"""
analyze_logs.py - Command-line log analysis dashboard for SecureBank.
Provides insights into system performance, error patterns, and usage statistics.
"""

import argparse
import sys
import os
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.utils.advanced_logging import LogAnalyzer, LogRotationManager


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}")


def print_metrics_table(metrics):
    """Print metrics in a formatted table."""
    print(f"{'Metric':<30} {'Value':<20}")
    print("-" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key:<30} {value:.3f}")
        elif isinstance(value, dict):
            print(f"{key:<30} [Dict with {len(value)} items]")
        else:
            print(f"{key:<30} {value}")


def analyze_system_performance(analyzer, hours_back):
    """Analyze and display system performance metrics."""
    print_section(f"SYSTEM PERFORMANCE ANALYSIS ({hours_back}h)")
    
    metrics = analyzer.get_system_metrics(hours_back=hours_back)
    
    # Overview
    print(f"Analysis Period: {hours_back} hours")
    print(f"Total Events: {metrics['total_events']:,}")
    print(f"Analysis Time: {metrics['analysis_timestamp']}")
    
    # Event breakdown
    print("\nEvent Types:")
    for event_type, count in metrics['event_breakdown'].items():
        percentage = (count / metrics['total_events'] * 100) if metrics['total_events'] > 0 else 0
        print(f"  {event_type:<25} {count:>6} ({percentage:5.1f}%)")
    
    # Performance metrics
    print("\nPerformance Metrics:")
    perf = metrics['performance_metrics']
    print_metrics_table(perf)
    
    # Error analysis
    print("\nError Analysis:")
    error = metrics['error_metrics']
    print_metrics_table(error)
    
    # Fraud detection
    print("\nFraud Detection:")
    fraud = metrics['fraud_detection_metrics']
    print_metrics_table(fraud)
    
    # Endpoint usage
    print("\nEndpoint Usage:")
    for endpoint, count in metrics['endpoint_usage'].items():
        percentage = (count / metrics['total_events'] * 100) if metrics['total_events'] > 0 else 0
        print(f"  {endpoint:<25} {count:>6} ({percentage:5.1f}%)")


def analyze_model_history(analyzer):
    """Analyze model training history."""
    print_section("MODEL TRAINING HISTORY")
    
    history = analyzer.get_model_performance_history()
    
    if not history:
        print("No model training sessions found in logs.")
        return
    
    print(f"Total Training Sessions: {len(history)}")
    print()
    
    # Show recent sessions
    print("Recent Training Sessions:")
    print(f"{'Date':<20} {'Model ID':<35} {'Precision':<10} {'Recall':<10} {'F1':<10} {'Time(s)':<10}")
    print("-" * 105)
    
    for session in history[-10:]:  # Last 10 sessions
        timestamp = session.get('timestamp', 'Unknown')
        if timestamp != 'Unknown':
            date_str = datetime.fromisoformat(timestamp).strftime('%Y-%m-%d %H:%M')
        else:
            date_str = 'Unknown'
        
        model_id = session.get('model_id', 'Unknown')[:34]  # Truncate long IDs
        precision = session.get('precision', 0)
        recall = session.get('recall', 0)
        f1 = session.get('f1_score', 0)
        training_time = session.get('training_time', 0)
        
        print(f"{date_str:<20} {model_id:<35} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f} {training_time:<10.1f}")
    
    # Performance trends
    if len(history) > 1:
        print("\nPerformance Trends:")
        recent_precision = [s.get('precision', 0) for s in history[-5:] if s.get('precision')]
        recent_recall = [s.get('recall', 0) for s in history[-5:] if s.get('recall')]
        
        if recent_precision:
            avg_precision = sum(recent_precision) / len(recent_precision)
            print(f"  Average Precision (last 5): {avg_precision:.3f}")
        
        if recent_recall:
            avg_recall = sum(recent_recall) / len(recent_recall)
            print(f"  Average Recall (last 5): {avg_recall:.3f}")


def analyze_daily_activity(analyzer, date_str=None):
    """Analyze daily activity patterns."""
    if date_str:
        try:
            date_obj = datetime.strptime(date_str, '%Y-%m-%d')
            date_title = date_str
        except ValueError:
            print(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")
            return
    else:
        date_obj = None
        date_title = "TODAY"
    
    print_section(f"DAILY ACTIVITY ANALYSIS - {date_title}")
    
    report = analyzer.generate_daily_report(date=date_obj)
    
    print(f"Report Date: {report['report_date']}")
    print(f"Total Activity: {report['total_activity']:,} events")
    
    if report['peak_hour'] is not None:
        print(f"Peak Activity Hour: {report['peak_hour']:02d}:00")
    
    # Hourly breakdown
    print("\nHourly Activity Distribution:")
    hourly = report['hourly_breakdown']
    if hourly:
        max_activity = max(hourly.values())
        print("Hour  Activity  " + "Chart")
        print("-" * 40)
        for hour in range(24):
            activity = hourly.get(hour, 0)
            bar_length = int((activity / max_activity) * 20) if max_activity > 0 else 0
            bar = "█" * bar_length
            print(f"{hour:02d}:00 {activity:>8}  {bar}")
    else:
        print("No activity recorded for this day.")
    
    # Summary stats
    print(f"\nDaily Summary:")
    print(f"  Model Training Sessions: {report['model_training_sessions']}")
    print(f"  Dataset Generations: {report['dataset_generations']}")


def show_log_directory_status(rotation_manager):
    """Show log directory status and cleanup recommendations."""
    print_section("LOG DIRECTORY STATUS")
    
    stats = rotation_manager.get_log_directory_stats()
    
    print(f"Total Log Files: {stats['total_files']:,}")
    print(f"Total Size: {stats['total_size_mb']} MB")
    
    if stats['oldest_file']:
        print(f"Oldest File: {stats['oldest_file']['age_days']} days old")
        print(f"  Path: {os.path.basename(stats['oldest_file']['path'])}")
    
    if stats['newest_file']:
        print(f"Newest File: {stats['newest_file']['age_hours']:.1f} hours old")
        print(f"  Path: {os.path.basename(stats['newest_file']['path'])}")
    
    # Cleanup recommendations
    print("\nCleanup Recommendations:")
    if stats['total_files'] > 500:
        print(f"  ⚠️  Large number of log files ({stats['total_files']})")
        print("     Consider running cleanup to remove old files")
    
    if stats['total_size_mb'] > 100:
        print(f"  ⚠️  Log directory using {stats['total_size_mb']} MB of disk space")
        print("     Consider log rotation if disk space is limited")
    
    if stats.get('oldest_file', {}).get('age_days', 0) > 90:
        print("  ⚠️  Very old log files detected (>90 days)")
        print("     Consider setting up automatic cleanup")


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="SecureBank Log Analysis Dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_logs.py --performance 24     # Analyze last 24 hours
  python analyze_logs.py --models             # Show model training history
  python analyze_logs.py --daily 2024-01-15  # Analyze specific date
  python analyze_logs.py --status             # Show log directory status
  python analyze_logs.py --all                # Run all analyses
        """
    )
    
    parser.add_argument('--performance', type=int, metavar='HOURS',
                       help='Analyze system performance for last N hours')
    parser.add_argument('--models', action='store_true',
                       help='Show model training history')
    parser.add_argument('--daily', metavar='DATE',
                       help='Analyze daily activity (YYYY-MM-DD format, or "today")')
    parser.add_argument('--status', action='store_true',
                       help='Show log directory status')
    parser.add_argument('--all', action='store_true',
                       help='Run all analyses')
    parser.add_argument('--logs-dir', default='logs',
                       help='Log directory path (default: logs)')
    
    args = parser.parse_args()
    
    # Initialize analyzers
    analyzer = LogAnalyzer(logs_dir=args.logs_dir)
    rotation_manager = LogRotationManager(logs_dir=args.logs_dir)
    
    # Check if log directory exists
    if not os.path.exists(args.logs_dir):
        print(f"Error: Log directory '{args.logs_dir}' not found.")
        print("Make sure the SecureBank system has been running and generating logs.")
        sys.exit(1)
    
    # Run requested analyses
    if args.all:
        analyze_system_performance(analyzer, 24)
        analyze_model_history(analyzer)
        analyze_daily_activity(analyzer)
        show_log_directory_status(rotation_manager)
    else:
        if args.performance:
            analyze_system_performance(analyzer, args.performance)
        
        if args.models:
            analyze_model_history(analyzer)
        
        if args.daily:
            date_str = None if args.daily.lower() == 'today' else args.daily
            analyze_daily_activity(analyzer, date_str)
        
        if args.status:
            show_log_directory_status(rotation_manager)
        
        # If no specific analysis requested, show help
        if not any([args.performance, args.models, args.daily, args.status]):
            parser.print_help()
            print("\nQuick start: python analyze_logs.py --all")


if __name__ == "__main__":
    main()