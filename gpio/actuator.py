"""
gpio/actuator.py — ASHAN
==========================
Controls the red and green LEDs via GPIO pins.
Stubs out gracefully when USE_PI_HARDWARE = False (laptop dev).

Wiring:
  GPIO 17 → 330Ω resistor → Red LED  → GND   (IN_USE  / Do Not Disturb)
  GPIO 27 → 330Ω resistor → Green LED → GND  (EMPTY   / Available)
"""

import logging

log = logging.getLogger(__name__)

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import app_config

_gpio_ready = False


def _init_gpio():
    global _gpio_ready
    if not app_config.USE_PI_HARDWARE:
        log.info("GPIO: running in stub mode (USE_PI_HARDWARE=False).")
        return

    try:
        import RPi.GPIO as GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(app_config.GPIO_RED_LED,   GPIO.OUT, initial=GPIO.LOW)
        GPIO.setup(app_config.GPIO_GREEN_LED, GPIO.OUT, initial=GPIO.LOW)
        _gpio_ready = True
        log.info("GPIO initialised.")
    except Exception as e:
        log.error(f"GPIO init failed: {e}")


def _set_pins(red: bool, green: bool):
    if not app_config.USE_PI_HARDWARE:
        log.info(f"GPIO stub → red={'ON' if red else 'OFF'}, green={'ON' if green else 'OFF'}")
        return
    try:
        import RPi.GPIO as GPIO
        GPIO.output(app_config.GPIO_RED_LED,   GPIO.HIGH if red   else GPIO.LOW)
        GPIO.output(app_config.GPIO_GREEN_LED, GPIO.HIGH if green else GPIO.LOW)
    except Exception as e:
        log.error(f"GPIO write error: {e}")


def set_room_state(state: str):
    """
    Set LEDs based on room state.
      IN_USE → red ON,  green OFF
      EMPTY  → red OFF, green ON
    """
    if not _gpio_ready and app_config.USE_PI_HARDWARE:
        _init_gpio()

    if state == "IN_USE":
        _set_pins(red=True,  green=False)
    elif state == "EMPTY":
        _set_pins(red=False, green=True)


def maintenance_lock():
    """Both LEDs on = maintenance mode."""
    _set_pins(red=True, green=True)
    log.info("Maintenance lock: both LEDs on.")


def release_lock():
    """Release maintenance lock — go back to EMPTY state."""
    _set_pins(red=False, green=True)
    log.info("Maintenance lock released.")


# Initialise on import
_init_gpio()
