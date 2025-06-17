"""
Extended tests for resource monitoring functionality to improve coverage
"""

import platform
import subprocess
import threading
import time
from unittest.mock import MagicMock, Mock, patch

import pytest

from mlx_websockets.resource_monitor import (
    ResourceMonitor,
    check_memory_available,
    get_thermal_state,
    suggest_memory_optimizations,
)


class TestResourceMonitorCore:
    """Test core ResourceMonitor functionality"""

    def test_init_defaults(self):
        """Test ResourceMonitor initialization with defaults"""
        monitor = ResourceMonitor()

        assert monitor.low_memory_threshold == 0.15
        assert monitor.critical_memory_threshold == 0.05
        assert monitor.check_interval == 30.0
        assert monitor.enable_battery_monitoring is True
        assert monitor._monitoring is False
        assert monitor._monitor_thread is None
        assert monitor._memory_warned is False
        assert monitor._battery_warned is False
        assert monitor._last_memory_check == 0.0
        assert monitor._last_battery_check == 0.0

    def test_init_custom(self):
        """Test ResourceMonitor initialization with custom values"""
        monitor = ResourceMonitor(
            low_memory_threshold=0.25,
            critical_memory_threshold=0.08,
            check_interval=10.0,
            enable_battery_monitoring=False,
        )

        assert monitor.low_memory_threshold == 0.25
        assert monitor.critical_memory_threshold == 0.08
        assert monitor.check_interval == 10.0
        assert monitor.enable_battery_monitoring is False

    def test_start_monitoring(self):
        """Test starting resource monitoring"""
        monitor = ResourceMonitor(check_interval=0.1)

        # Not monitoring initially
        assert monitor._monitoring is False
        assert monitor._monitor_thread is None

        # Start monitoring
        monitor.start()

        assert monitor._monitoring is True
        assert monitor._monitor_thread is not None
        assert monitor._monitor_thread.is_alive()
        assert monitor._monitor_thread.daemon is True
        assert monitor._monitor_thread.name == "ResourceMonitor"

        # Stop monitoring
        monitor.stop()
        time.sleep(0.2)  # Give thread time to stop

        assert monitor._monitoring is False
        # Thread should have stopped
        assert not monitor._monitor_thread.is_alive()

    def test_start_when_already_running(self):
        """Test starting monitor when already running"""
        monitor = ResourceMonitor(check_interval=0.1)

        # Start monitoring
        monitor.start()
        first_thread = monitor._monitor_thread

        # Try to start again
        monitor.start()
        second_thread = monitor._monitor_thread

        # Should be the same thread
        assert first_thread is second_thread

        monitor.stop()

    def test_stop_when_not_running(self):
        """Test stopping monitor when not running"""
        monitor = ResourceMonitor()

        # Should not raise exception
        monitor.stop()
        assert monitor._monitoring is False

    def test_stop_with_timeout(self):
        """Test stopping monitor with thread join timeout"""
        monitor = ResourceMonitor(check_interval=10.0)  # Long interval

        monitor.start()
        assert monitor._monitor_thread.is_alive()

        # Stop should work even with long check interval
        monitor.stop()

        # Thread should stop within timeout
        assert monitor._monitoring is False


class TestMonitoringLoop:
    """Test the monitoring loop functionality"""

    @patch("mlx_websockets.resource_monitor.logger")
    def test_monitor_loop_exception_handling(self, mock_logger):
        """Test that monitoring loop handles exceptions"""
        monitor = ResourceMonitor(check_interval=0.1)

        # Mock check methods to raise exceptions
        monitor._check_memory = Mock(side_effect=Exception("Memory check failed"))
        monitor._check_battery = Mock(side_effect=Exception("Battery check failed"))

        # Start monitoring
        monitor.start()

        # Let it run a few cycles
        time.sleep(0.3)

        # Should have logged errors but kept running
        assert mock_logger.error.called
        assert monitor._monitor_thread.is_alive()

        # Stop monitoring
        monitor.stop()

    def test_monitor_loop_memory_checks(self):
        """Test that monitoring loop performs memory checks"""
        monitor = ResourceMonitor(check_interval=0.1)

        # Mock check methods
        monitor._check_memory = Mock()
        monitor._check_battery = Mock()

        # Start monitoring
        monitor.start()

        # Let it run a few cycles
        time.sleep(0.3)

        # Should have called check_memory multiple times
        assert monitor._check_memory.call_count >= 2

        # Stop monitoring
        monitor.stop()

    def test_monitor_loop_battery_checks(self):
        """Test that monitoring loop performs battery checks"""
        monitor = ResourceMonitor(check_interval=0.1, enable_battery_monitoring=True)

        # Mock check methods
        monitor._check_memory = Mock()
        monitor._check_battery = Mock()

        # Start monitoring
        monitor.start()

        # Battery checks happen every 60 seconds, so wait briefly
        time.sleep(0.2)

        # Battery check might not have been called yet (60s interval)
        # Force a battery check by manipulating the timestamp
        monitor._last_battery_check = time.time() - 61
        time.sleep(0.2)

        # Stop monitoring
        monitor.stop()


class TestMemoryChecking:
    """Test memory checking functionality"""

    @patch("psutil.virtual_memory")
    @patch("mlx_websockets.resource_monitor.log_error")
    def test_check_memory_critical(self, mock_log_error, mock_vm):
        """Test critical memory warning"""
        mock_vm.return_value = MagicMock(
            total=16 * 1024**3,  # 16GB
            available=0.5 * 1024**3,  # 0.5GB (3.125%)
        )

        monitor = ResourceMonitor(critical_memory_threshold=0.05)
        monitor._memory_warned = False

        monitor._check_memory()

        # Should have logged critical error
        mock_log_error.assert_called_once()
        assert "CRITICAL" in str(mock_log_error.call_args)
        assert monitor._memory_warned is True

    @patch("psutil.virtual_memory")
    @patch("mlx_websockets.resource_monitor.log_warning")
    def test_check_memory_low(self, mock_log_warning, mock_vm):
        """Test low memory warning"""
        mock_vm.return_value = MagicMock(
            total=16 * 1024**3,  # 16GB
            available=2 * 1024**3,  # 2GB (12.5%)
        )

        monitor = ResourceMonitor(low_memory_threshold=0.15, critical_memory_threshold=0.05)
        monitor._memory_warned = False

        monitor._check_memory()

        # Should have logged warning
        mock_log_warning.assert_called_once()
        assert "WARNING" in str(mock_log_warning.call_args)
        assert monitor._memory_warned is True

    @patch("psutil.virtual_memory")
    @patch("mlx_websockets.resource_monitor.mx")
    @patch("mlx_websockets.resource_monitor.log_error")
    def test_check_memory_with_mlx(self, mock_log_error, mock_mx, mock_vm):
        """Test memory check including MLX memory"""
        mock_vm.return_value = MagicMock(
            total=16 * 1024**3,
            available=0.5 * 1024**3,
        )
        mock_mx.get_active_memory.return_value = 4 * 1024**3  # 4GB MLX

        monitor = ResourceMonitor()
        monitor._memory_warned = False

        monitor._check_memory()

        # Should include MLX memory in warning
        mock_log_error.assert_called_once()
        call_str = str(mock_log_error.call_args)
        assert "MLX using 4.0GB" in call_str

    @patch("psutil.virtual_memory")
    def test_check_memory_reset_warning(self, mock_vm):
        """Test memory warning flag resets when memory is OK"""
        mock_vm.return_value = MagicMock(
            total=16 * 1024**3,
            available=8 * 1024**3,  # 50% free
        )

        monitor = ResourceMonitor()
        monitor._memory_warned = True  # Was previously warned

        monitor._check_memory()

        # Warning flag should be reset
        assert monitor._memory_warned is False

    @patch("psutil.virtual_memory")
    @patch("mlx_websockets.resource_monitor.logger")
    def test_check_memory_exception(self, mock_logger, mock_vm):
        """Test memory check handles exceptions"""
        mock_vm.side_effect = Exception("Memory check failed")

        monitor = ResourceMonitor()

        # Should not raise exception
        monitor._check_memory()

        # Should log debug message
        mock_logger.debug.assert_called()


class TestBatteryChecking:
    """Test battery checking functionality"""

    @patch("platform.system")
    def test_check_battery_non_darwin(self, mock_platform):
        """Test battery check on non-macOS systems"""
        mock_platform.return_value = "Linux"

        monitor = ResourceMonitor()
        monitor._check_battery()

        # Should return early without doing anything

    @patch("platform.system")
    @patch("psutil.sensors_battery")
    def test_check_battery_no_battery(self, mock_battery, mock_platform):
        """Test battery check when no battery present"""
        mock_platform.return_value = "Darwin"
        mock_battery.return_value = None

        monitor = ResourceMonitor()
        monitor._check_battery()

        # Should handle gracefully

    @patch("platform.system")
    @patch("psutil.sensors_battery")
    @patch("mlx_websockets.resource_monitor.log_warning")
    def test_check_battery_low_unplugged(self, mock_log_warning, mock_battery, mock_platform):
        """Test low battery warning when unplugged"""
        mock_platform.return_value = "Darwin"
        mock_battery.return_value = MagicMock(percent=15, power_plugged=False)

        monitor = ResourceMonitor()
        monitor._battery_warned = False

        monitor._check_battery()

        # Should warn about low battery
        mock_log_warning.assert_called_once()
        assert "battery power (15%)" in str(mock_log_warning.call_args)
        assert monitor._battery_warned is True

    @patch("platform.system")
    @patch("psutil.sensors_battery")
    def test_check_battery_plugged_in(self, mock_battery, mock_platform):
        """Test no warning when plugged in"""
        mock_platform.return_value = "Darwin"
        mock_battery.return_value = MagicMock(percent=15, power_plugged=True)

        monitor = ResourceMonitor()
        monitor._battery_warned = True  # Was previously warned

        monitor._check_battery()

        # Should reset warning flag
        assert monitor._battery_warned is False

    @patch("platform.system")
    @patch("psutil.sensors_battery")
    @patch("mlx_websockets.resource_monitor.logger")
    def test_check_battery_exception(self, mock_logger, mock_battery, mock_platform):
        """Test battery check handles exceptions"""
        mock_platform.return_value = "Darwin"
        mock_battery.side_effect = Exception("Battery check failed")

        monitor = ResourceMonitor()

        # Should not raise exception
        monitor._check_battery()

        # Should log debug message
        mock_logger.debug.assert_called()


class TestResourceStats:
    """Test resource statistics gathering"""

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    def test_get_resource_stats_basic(self, mock_cpu_count, mock_cpu_percent, mock_vm):
        """Test basic resource stats"""
        mock_vm.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3, percent=50.0)
        mock_cpu_percent.return_value = 25.0
        mock_cpu_count.return_value = 8

        monitor = ResourceMonitor()
        stats = monitor.get_resource_stats()

        assert stats["memory"]["total_gb"] == 16.0
        assert stats["memory"]["available_gb"] == 8.0
        assert stats["memory"]["percent_used"] == 50.0
        assert stats["memory"]["percent_free"] == 50.0
        assert stats["cpu"]["percent"] == 25.0
        assert stats["cpu"]["count"] == 8

    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    @patch("mlx_websockets.resource_monitor.mx")
    def test_get_resource_stats_with_mlx(self, mock_mx, mock_cpu_count, mock_cpu_percent, mock_vm):
        """Test resource stats with MLX memory"""
        mock_vm.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3, percent=50.0)
        mock_cpu_percent.return_value = 25.0
        mock_cpu_count.return_value = 8
        mock_mx.get_active_memory.return_value = 2 * 1024**3

        monitor = ResourceMonitor()
        stats = monitor.get_resource_stats()

        assert stats["mlx_memory_gb"] == 2.0

    @patch("platform.system")
    @patch("psutil.virtual_memory")
    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    @patch("psutil.sensors_battery")
    def test_get_resource_stats_with_battery(
        self, mock_battery, mock_cpu_count, mock_cpu_percent, mock_vm, mock_platform
    ):
        """Test resource stats with battery info on macOS"""
        mock_platform.return_value = "Darwin"
        mock_vm.return_value = MagicMock(total=16 * 1024**3, available=8 * 1024**3, percent=50.0)
        mock_cpu_percent.return_value = 25.0
        mock_cpu_count.return_value = 8
        mock_battery.return_value = MagicMock(percent=75.0, power_plugged=True, secsleft=3600)

        monitor = ResourceMonitor()
        stats = monitor.get_resource_stats()

        assert stats["battery"]["percent"] == 75.0
        assert stats["battery"]["plugged"] is True
        assert stats["battery"]["time_remaining"] == 3600

    @patch("psutil.virtual_memory")
    @patch("mlx_websockets.resource_monitor.logger")
    def test_get_resource_stats_exception(self, mock_logger, mock_vm):
        """Test resource stats handles exceptions"""
        mock_vm.side_effect = Exception("Stats failed")

        monitor = ResourceMonitor()
        stats = monitor.get_resource_stats()

        # Should return empty dict on error
        assert stats == {}
        mock_logger.error.assert_called()


class TestUtilityFunctions:
    """Test utility functions"""

    @patch("psutil.virtual_memory")
    def test_check_memory_available_sufficient(self, mock_vm):
        """Test memory check with sufficient memory"""
        mock_vm.return_value = MagicMock(available=8 * 1024**3)

        ok, msg = check_memory_available(required_gb=4.0)

        assert ok is True
        assert "8.0GB available" in msg

    @patch("psutil.virtual_memory")
    def test_check_memory_available_insufficient(self, mock_vm):
        """Test memory check with insufficient memory"""
        mock_vm.return_value = MagicMock(available=2 * 1024**3)

        ok, msg = check_memory_available(required_gb=4.0)

        assert ok is False
        assert "2.0GB available" in msg
        assert "4.0GB required" in msg

    @patch("psutil.virtual_memory")
    @patch("mlx_websockets.resource_monitor.logger")
    def test_check_memory_available_exception(self, mock_logger, mock_vm):
        """Test memory check handles exceptions"""
        mock_vm.side_effect = Exception("Memory error")

        ok, msg = check_memory_available(required_gb=4.0)

        # Should default to True on error
        assert ok is True
        assert "Could not check memory" in msg
        mock_logger.error.assert_called()

    @patch("platform.system")
    def test_get_thermal_state_non_darwin(self, mock_platform):
        """Test thermal state on non-macOS"""
        mock_platform.return_value = "Linux"

        state = get_thermal_state()

        assert state is None

    @patch("platform.system")
    @patch("subprocess.run")
    def test_get_thermal_state_throttled(self, mock_run, mock_platform):
        """Test thermal state when throttled"""
        mock_platform.return_value = "Darwin"
        mock_run.return_value = MagicMock(returncode=0, stdout="CPU_Speed_Limit = 50%")

        state = get_thermal_state()

        assert state == "throttled"

    @patch("platform.system")
    @patch("subprocess.run")
    def test_get_thermal_state_normal(self, mock_run, mock_platform):
        """Test thermal state when normal"""
        mock_platform.return_value = "Darwin"
        mock_run.return_value = MagicMock(returncode=0, stdout="Normal thermal state")

        state = get_thermal_state()

        assert state == "normal"

    @patch("platform.system")
    @patch("subprocess.run")
    def test_get_thermal_state_exception(self, mock_run, mock_platform):
        """Test thermal state handles exceptions"""
        mock_platform.return_value = "Darwin"
        mock_run.side_effect = Exception("Command failed")

        state = get_thermal_state()

        assert state is None

    def test_suggest_memory_optimizations_high_usage(self):
        """Test memory optimization suggestions for high usage"""
        suggestions = suggest_memory_optimizations(mlx_memory_gb=10.0)

        assert len(suggestions) >= 2
        assert any("smaller model" in s for s in suggestions)
        assert any("reducing max_tokens" in s for s in suggestions)

    def test_suggest_memory_optimizations_medium_usage(self):
        """Test memory optimization suggestions for medium usage"""
        suggestions = suggest_memory_optimizations(mlx_memory_gb=6.0)

        assert len(suggestions) >= 2
        assert any("Close other applications" in s for s in suggestions)

    @patch("psutil.virtual_memory")
    def test_suggest_memory_optimizations_critical_system(self, mock_vm):
        """Test suggestions when system memory is critical"""
        mock_vm.return_value = MagicMock(percent=92.0)

        suggestions = suggest_memory_optimizations(mlx_memory_gb=2.0)

        assert any("critically low" in s for s in suggestions)
        assert any("restart" in s for s in suggestions)


class TestThreadSafety:
    """Test thread safety aspects"""

    def test_concurrent_start_stop(self):
        """Test concurrent start/stop operations"""
        monitor = ResourceMonitor(check_interval=0.1)

        # Define functions to run in threads
        def start_monitor():
            for _ in range(5):
                monitor.start()
                time.sleep(0.01)

        def stop_monitor():
            for _ in range(5):
                monitor.stop()
                time.sleep(0.01)

        # Run start/stop concurrently
        threads = []
        for _ in range(3):
            threads.append(threading.Thread(target=start_monitor))
            threads.append(threading.Thread(target=stop_monitor))

        for t in threads:
            t.start()

        for t in threads:
            t.join()

        # Should end in a stable state
        monitor.stop()
        assert not monitor._monitoring

    def test_get_stats_while_monitoring(self):
        """Test getting stats while monitoring is running"""
        monitor = ResourceMonitor(check_interval=0.1)

        monitor.start()

        # Get stats multiple times while monitoring
        stats_list = []
        for _ in range(5):
            stats = monitor.get_resource_stats()
            stats_list.append(stats)
            time.sleep(0.05)

        monitor.stop()

        # All stats should be valid
        for stats in stats_list:
            assert "memory" in stats
            assert "cpu" in stats
