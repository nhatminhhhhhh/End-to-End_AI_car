import time

PCA9685_DEFAULT_ADDR = 0x40
MODE1 = 0x00
PRESCALE = 0xFE
LED0_ON_L = 0x06


class PCA9685:
    def __init__(self, bus=0, address=PCA9685_DEFAULT_ADDR, freq=50):
        self.address = address
        self.freq = freq

        try:
            from smbus2 import SMBus
        except ImportError:
            from smbus import SMBus

        self.bus = SMBus(bus)
        self.reset()
        self.set_pwm_freq(freq)

    def reset(self):
        self.bus.write_byte_data(self.address, MODE1, 0x00)
        time.sleep(0.01)

    def set_pwm_freq(self, freq):
        prescale = int(round(25e6 / (4096 * freq)) - 1)
        old_mode = self.bus.read_byte_data(self.address, MODE1)
        self.bus.write_byte_data(self.address, MODE1, (old_mode & 0x7F) | 0x10)
        self.bus.write_byte_data(self.address, PRESCALE, prescale)
        self.bus.write_byte_data(self.address, MODE1, old_mode)
        time.sleep(0.005)
        self.bus.write_byte_data(self.address, MODE1, old_mode | 0xA0)

    def set_pwm(self, channel, on, off):
        reg = LED0_ON_L + 4 * channel
        self.bus.write_byte_data(self.address, reg, on & 0xFF)
        self.bus.write_byte_data(self.address, reg + 1, (on >> 8) & 0xFF)
        self.bus.write_byte_data(self.address, reg + 2, off & 0xFF)
        self.bus.write_byte_data(self.address, reg + 3, (off >> 8) & 0xFF)

    def set_servo_pulse(self, channel, pulse_ms):
        period = 1000.0 / self.freq
        ticks = int(4096 * pulse_ms / period)
        ticks = max(0, min(4095, ticks))
        self.set_pwm(channel, 0, ticks)

    def angle_to_pulse(self, angle):
        SERVO_PULSE_MIN = 0.5
        SERVO_PULSE_MAX = 2.5
        angle = max(0, min(180, angle))
        return SERVO_PULSE_MIN + (angle / 180.0) * (SERVO_PULSE_MAX - SERVO_PULSE_MIN)
