from threading import Lock
from typing import Protocol, Sequence
import time

import numpy as np
from dynamixel_sdk.group_sync_read import GroupSyncRead
from dynamixel_sdk.group_sync_write import GroupSyncWrite
from dynamixel_sdk.packet_handler import PacketHandler
from dynamixel_sdk.port_handler import PortHandler
from dynamixel_sdk.robotis_def import (
    COMM_SUCCESS,
    DXL_HIBYTE,
    DXL_HIWORD,
    DXL_LOBYTE,
    DXL_LOWORD,
)

ADDR_TORQUE_ENABLE = 64
ADDR_GOAL_CURRENT = 102
LEN_GOAL_CURRENT = 2
ADDR_PRESENT_POSITION = 132
LEN_PRESENT_POSITION = 4
ADDR_PRESENT_VELOCITY = 128
LEN_PRESENT_VELOCITY = 4
ADDR_GOAL_POSITION = 116
LEN_GOAL_POSITION = 4
TORQUE_ENABLE = 1
TORQUE_DISABLE = 0
ADDR_OPERATING_MODE = 11
CURRENT_CONTROL_MODE = 0
POSITION_CONTROL_MODE = 3

TORQUE_TO_CURRENT_MAPPING = {
    "XC330_T288_T": 1158.73,
    "XM430_W210_T": 1000/2.69,
}

class DynamixelDriverProtocol(Protocol):
    def set_current(self, currents: Sequence[float]):
        """Set the current for the Dynamixel servos.

        Args:
            currents (Sequence[float]): A list of currents in mA.
        """
        ...

    def torque_enabled(self) -> bool:
        """Check if torque is enabled for the Dynamixel servos.

        Returns:
            bool: True if torque is enabled, False if it is disabled.
        """
        ...

    def set_torque_mode(self, enable: bool):
        """Set the torque mode for the Dynamixel servos.

        Args:
            enable (bool): True to enable torque, False to disable.
        """
        ...

    def get_positions(self) -> np.ndarray:
        """Get the current joint angles in radians.

        Returns:
            np.ndarray: An array of joint angles.
        """
        ...

    def close(self):
        """Close the driver."""


class DynamixelDriver(DynamixelDriverProtocol):
    def __init__(self, ids: Sequence[int], servo_types: Sequence[str], port: str = "/dev/ttyUSB0", baudrate: int = 4000000):
        self._ids = ids
        self._positions = None
        self._lock = Lock()
        
        self._stopped = False

        self._portHandler = PortHandler(port)
        self._packetHandler = PacketHandler(2.0)
        self._groupSyncRead = GroupSyncRead(
            self._portHandler, self._packetHandler, ADDR_PRESENT_VELOCITY, LEN_PRESENT_POSITION + LEN_PRESENT_VELOCITY,
        )
        self._groupSyncWrite = GroupSyncWrite(
            self._portHandler, self._packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT,
        )
        if not self._portHandler.openPort():
            raise RuntimeError("Failed to open the port")
        if not self._portHandler.setBaudRate(baudrate):
            raise RuntimeError(f"Failed to change the baudrate, {baudrate}")

        for dxl_id in self._ids:
            if not self._groupSyncRead.addParam(dxl_id):
                raise RuntimeError(f"Failed to add parameter for Dynamixel with ID {dxl_id}")
        
        self.torque_to_current_map = np.array(
            [TORQUE_TO_CURRENT_MAPPING[servo] for servo in servo_types]
        )

        self._torque_enabled = False
        try:
            self.set_torque_mode(self._torque_enabled)
        except Exception as e:
            print(f"port: {port}, {e}")

    @property
    def torque_enabled(self) -> bool:
        return self._torque_enabled

    def safe_disable(self, verify_timeout_s: float = 0.30):
        with self._lock:
            # 0) Tell all call sites to stop using the bus immediately
            self._stopped = True

            # 1) Best-effort zero current (prevents hold even if TE briefly lingers)
            try:
                gsw_cur = GroupSyncWrite(self._portHandler, self._packetHandler, ADDR_GOAL_CURRENT, LEN_GOAL_CURRENT)
                for dxl_id in self._ids:
                    gsw_cur.addParam(dxl_id, [0x00, 0x00])
                gsw_cur.txPacket()
                gsw_cur.clearParam()
            except Exception:
                pass

            # 1b) Best-effort zero PWM (addr 100, len 2) — harmless if unsupported
            try:
                gsw_pwm = GroupSyncWrite(self._portHandler, self._packetHandler, 100, 2)
                for dxl_id in self._ids:
                    gsw_pwm.addParam(dxl_id, [0x00, 0x00])
                gsw_pwm.txPacket()
                gsw_pwm.clearParam()
            except Exception:
                pass

            # 2) Torque off: (a) broadcast tx-only, then (b) atomic syncwrite
            try:
                if hasattr(self._packetHandler, "write1ByteTxOnly"):
                    # 0xFE = broadcast ID in Dynamixel protocol 2.0
                    self._packetHandler.write1ByteTxOnly(self._portHandler, 0xFE, ADDR_TORQUE_ENABLE, 0x00)
            except Exception:
                pass  # broadcast is best-effort; we'll still syncwrite below

            try:
                gsw = GroupSyncWrite(self._portHandler, self._packetHandler, ADDR_TORQUE_ENABLE, 1)
                for dxl_id in self._ids:
                    gsw.addParam(dxl_id, [0x00])
                gsw.txPacket()
                gsw.clearParam()
            finally:
                # 3) Let USB/TTL drain so close() won't eat the last packet
                time.sleep(0.05)

            self._torque_enabled = False

            # 4) Verify for a short window; retry any sticky IDs per-ID
            deadline = time.time() + verify_timeout_s
            bad = set()
            while time.time() < deadline:
                bad.clear()
                for dxl_id in self._ids:
                    val, res, err = self._packetHandler.read1ByteTxRx(self._portHandler, dxl_id, ADDR_TORQUE_ENABLE)
                    if res != COMM_SUCCESS or err != 0 or val != 0:
                        bad.add(dxl_id)
                if not bad:
                    break
                for dxl_id in list(bad):
                    try:
                        self._packetHandler.write1ByteTxRx(self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, 0x00)
                    except Exception:
                        pass
                time.sleep(0.01)

            if bad:
                print(f"[WARN] TorqueEnable still 1 for IDs: {sorted(bad)} (continuing shutdown)")

    def set_torque_mode(self, enable: bool):
        with self._lock:
            errs = []
            for dxl_id in self._ids:
                res, err = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, TORQUE_ENABLE if enable else TORQUE_DISABLE
                )
                if res != COMM_SUCCESS or err != 0:
                    errs.append(dxl_id)
            # Do NOT bail early; attempt all IDs
            if errs:
                raise RuntimeError(f"Failed torque {'enable' if enable else 'disable'} for IDs {errs}")
            self._torque_enabled = enable
    
    def safe_set_torque_mode(
        self,
        enable: bool,
        *,
        verify_timeout_s: float = 0.30,
        sequential: bool = True,
        interprocess_lock_path: str | None = None,  # e.g. "/tmp/gello_torque.lock"
    ) -> bool:
        """
        Robustly set TorqueEnable (ADDR 64) for all IDs with verification.

        Args:
            enable: True -> TE=1, False -> TE=0
            verify_timeout_s: total time to verify & retry stragglers
            sequential: if True, per-ID enable/disable (gentler on power/TTL)
                        if False, one GroupSyncWrite then verify
            interprocess_lock_path: optional OS-wide file lock to avoid two nodes
                                    toggling torque at the exact same time

        Returns:
            bool: True if all IDs reached desired TE state within timeout; else False.
        """
        # --- optional cross-process lock (best-effort) ---
        lock_cm = getattr(self, "_file_lock", None)
        if lock_cm is None:
            # lightweight inline fallback if you didn't wire a helper
            import contextlib, os
            try:
                import fcntl
                _HAVE_FCNTL = True
            except Exception:
                _HAVE_FCNTL = False

            @contextlib.contextmanager
            def _file_lock(path: str | None):
                if not path or not _HAVE_FCNTL:
                    yield
                    return
                fd = None
                try:
                    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
                    fcntl.flock(fd, fcntl.LOCK_EX)
                    yield
                finally:
                    if fd is not None:
                        try:
                            fcntl.flock(fd, fcntl.LOCK_UN)
                        finally:
                            os.close(fd)
            lock_cm = _file_lock

        goal = TORQUE_ENABLE if enable else TORQUE_DISABLE

        with lock_cm(interprocess_lock_path):
            with self._lock:
                # Best-effort broadcast (won't raise)
                try:
                    if hasattr(self._packetHandler, "write1ByteTxOnly"):
                        self._packetHandler.write1ByteTxOnly(
                            self._portHandler, 0xFE, ADDR_TORQUE_ENABLE, goal
                        )
                except Exception:
                    pass

                # Primary write: sequential or syncwrite
                if sequential:
                    for dxl_id in self._ids:
                        try:
                            self._packetHandler.write1ByteTxRx(
                                self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, goal
                            )
                        except Exception:
                            pass
                        time.sleep(0.003)  # tiny stagger to be kind to the bus
                else:
                    try:
                        gsw = GroupSyncWrite(
                            self._portHandler, self._packetHandler, ADDR_TORQUE_ENABLE, 1
                        )
                        for dxl_id in self._ids:
                            gsw.addParam(dxl_id, [goal])
                        gsw.txPacket()
                        gsw.clearParam()
                    except Exception:
                        # fall back to sequential if syncwrite fails
                        for dxl_id in self._ids:
                            try:
                                self._packetHandler.write1ByteTxRx(
                                    self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, goal
                                )
                            except Exception:
                                pass

                # Let USB/TTL drain a hair before verify loop
                time.sleep(0.02)

                # Verify + repair loop
                deadline = time.time() + verify_timeout_s
                bad = set()
                while time.time() < deadline:
                    bad.clear()
                    for dxl_id in self._ids:
                        val, res, err = self._packetHandler.read1ByteTxRx(
                            self._portHandler, dxl_id, ADDR_TORQUE_ENABLE
                        )
                        if res != COMM_SUCCESS or err != 0 or val != goal:
                            bad.add(dxl_id)
                    if not bad:
                        break
                    for dxl_id in list(bad):
                        try:
                            self._packetHandler.write1ByteTxRx(
                                self._portHandler, dxl_id, ADDR_TORQUE_ENABLE, goal
                            )
                        except Exception:
                            pass
                    time.sleep(0.01)

                ok = len(bad) == 0
                self._torque_enabled = enable and ok
                if not ok:
                    print(f"[WARN] TorqueEnable != {goal} for IDs: {sorted(bad)}")
                return ok

    def close(self):
        with self._lock:
            try:
                self._portHandler.closePort()
            finally:
                self._stopped = True
                self._torque_enabled = False

    def set_operating_mode(self, mode: int):
        with self._lock:
            for dxl_id in self._ids:
                dxl_comm_result, dxl_error = self._packetHandler.write1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_OPERATING_MODE, mode
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0:
                    raise RuntimeError(f"Failed to set operating mode for Dynamixel with ID {dxl_id}")
    
    def verify_operating_mode(self, expected_mode: int):
        with self._lock:
            for dxl_id in self._ids:
                mode, dxl_comm_result, dxl_error = self._packetHandler.read1ByteTxRx(
                    self._portHandler, dxl_id, ADDR_OPERATING_MODE
                )
                if dxl_comm_result != COMM_SUCCESS or dxl_error != 0 or mode != expected_mode:
                    raise RuntimeError(f"Operating mode mismatch for Dynamixel ID {dxl_id}")
                
    def get_positions_and_velocities(self):
        # If we’re tearing down, don’t touch the bus
        if self._stopped:
            # Return the last cached positions (or zeros) and zeros for vel
            if self._positions is None:
                n = len(self._ids)
                return np.zeros(n), np.zeros(n)
            positions_in_radians = self._positions / 2048.0 * np.pi
            velocities_in_units = np.zeros_like(self._positions, dtype=float)
            return positions_in_radians, velocities_in_units

        with self._lock:
            _positions = np.zeros(len(self._ids), dtype=int)
            _velocities = np.zeros(len(self._ids), dtype=int)

            dxl_comm_result = self._groupSyncRead.txRxPacket()
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError(f"Warning, communication failed: {dxl_comm_result}")
            
            for i, dxl_id in enumerate(self._ids):
                if self._groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY):
                    velocity = self._groupSyncRead.getData(dxl_id, ADDR_PRESENT_VELOCITY, LEN_PRESENT_VELOCITY)
                    if velocity > 0x7FFFFFFF:
                        velocity -= 0x100000000
                    _velocities[i] = velocity
                else:
                    raise RuntimeError(f"Failed to get velocity for Dynamixel with ID {dxl_id}")

                if self._groupSyncRead.isAvailable(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION):
                    position = self._groupSyncRead.getData(dxl_id, ADDR_PRESENT_POSITION, LEN_PRESENT_POSITION)
                    if position > 0x7FFFFFFF:
                        position -= 0x100000000
                    _positions[i] = position
                else:
                    raise RuntimeError(f"Failed to get position for Dynamixel with ID {dxl_id}")

            self._positions = _positions
            self._velocities = _velocities

            positions_in_radians = _positions / 2048.0 * np.pi
            velocities_in_units = _velocities * 0.229 * 2 * np.pi / 60
            return positions_in_radians, velocities_in_units

    def set_current(self, currents: Sequence[float]):
        if self._stopped:
            return
        if len(currents) != len(self._ids):
            raise ValueError("currents length mismatch")
        with self._lock:
            if self._stopped or not self._torque_enabled:
                return  # ignore during/after shutdown
            currents = np.clip(currents, -900, 900)
            for dxl_id, current in zip(self._ids, currents):
                cv = int(current)
                if not self._groupSyncWrite.addParam(dxl_id, [DXL_LOBYTE(cv), DXL_HIBYTE(cv)]):
                    raise RuntimeError(f"Failed to stage current for ID {dxl_id}")
            res = self._groupSyncWrite.txPacket()
            self._groupSyncWrite.clearParam()
            if res != COMM_SUCCESS:
                raise RuntimeError("Failed to syncwrite goal current")

    def set_torque(self, torques: Sequence[float]):
        if self._stopped:
            return
        currents = self.torque_to_current_map*torques
        self.set_current(currents)

def main():
    # script for testing purposes
    ids = [1, 2, 3, 4, 5, 6, 7]
    port = "/dev/serial/by-id/usb-FTDI_USB__-__Serial_Converter_FT8ISV6J-if00-port0"

    try:
        driver = DynamixelDriver(ids, port=port, baudrate=4000000)
    except FileNotFoundError:
        print(f"Port {port} not found. Please check the connection.")
        return
    
    driver.set_operating_mode(0)
    driver.set_torque_mode(True)

    try:
        while True:
            positions = driver.get_positions()
            print(f"Current joint positions for IDs {ids}: {positions}")

            current_values = [0, 0, 0, 0, 0, 0, 0.0]
            driver.set_current(current_values)
    except KeyboardInterrupt:
        driver.set_torque_mode(False)
        driver.close()

if __name__ == "__main__":
    main()
