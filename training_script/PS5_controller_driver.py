import time
from dualsense_controller import (
    DeviceInfo, DualSenseController, Mapping, UpdateLevel,
    JoyStick, Number
)

class ps5_controller_driver:
    def __init__(self):
        self.is_running = True

        devices = DualSenseController.enumerate_devices()
        if not devices:
            raise Exception("No DualSense Controller found.")
        self.controller = DualSenseController(
            device_index_or_device_info=devices[0],
            mapping=Mapping.RAW,
            left_joystick_deadzone=5,
            right_joystick_deadzone=5,
            left_trigger_deadzone=1,
            right_trigger_deadzone=1,
            gyroscope_threshold=0,
            accelerometer_threshold=0,
            orientation_threshold=0,
            update_level=UpdateLevel.DEFAULT,
            microphone_initially_muted=True,
            microphone_invert_led=False,
        )

        #Set Blue Lightbar Color
        self.controller.lightbar.set_color_blue()

        # Add resistance levl for both triggers
        # self.controller.left_trigger.effect.no_resistance()
        # self.controller.left_trigger.effect.section_resistance()
        self.controller.left_trigger.effect.continuous_resistance()

        # self.controller.right_trigger.effect.no_resistance()
        # self.controller.right_trigger.effect.section_resistance()
        self.controller.right_trigger.effect.continuous_resistance()

        # State
        self.left_joystick = (0, 0)
        self.right_joystick = (0, 0)
        self.gripper_value = 0.0
        self.pressed_buttons = set()
        self.last_l2_value = 0
        self.last_r2_value = 0

        # Callbacks
        self.controller.btn_ps.on_down(self.on_btn_ps_down)
        self.controller.left_stick.on_change(self.on_left_stick_changed)
        self.controller.right_stick.on_change(self.on_right_stick_changed)
        self.controller.left_trigger.on_change(self.on_left_trigger_changed)
        self.controller.right_trigger.on_change(self.on_right_trigger_changed)

        ## Buttons to track
        self.controller.btn_triangle.on_change(lambda pressed: self._mark_button("triangle", pressed))

        self.controller.btn_cross.on_change(lambda pressed: self._mark_button("cross", pressed))

        self.controller.btn_circle.on_change(lambda pressed: self._mark_button("circle", pressed))

        self.controller.btn_square.on_change(lambda pressed: self._mark_button("square", pressed))

        self.controller.btn_left.on_change(lambda pressed: self._mark_button("left", pressed))
        
        self.controller.btn_up.on_change(lambda pressed: self._mark_button("up", pressed))

        self.controller.btn_right.on_change(lambda pressed: self._mark_button("right", pressed))

        self.controller.btn_down.on_change(lambda pressed: self._mark_button("down", pressed))

        self.controller.btn_l1.on_change(lambda pressed: self._mark_button("l1", pressed))
        self.controller.btn_r1.on_change(lambda pressed: self._mark_button("r1", pressed))

        self.controller.btn_l3.on_change(lambda pressed: self._mark_button("l3", pressed))
        self.controller.btn_r3.on_change(lambda pressed: self._mark_button("r3", pressed))


    #Update State of button on press and release
    def _mark_button(self, button_name: str, pressed: bool = True):
        if pressed:
            self.pressed_buttons.add(button_name)
        else:
            self.pressed_buttons.discard(button_name)

    def on_btn_ps_down(self):
        print("PS button pressed -> Exiting")
        self.is_running = False

    #Scale value of 0 -> 255 to -1.0 to 1.0 
    def normalize(self, value, deadzone=5):
        mid = 127.5
        val = (value - mid) / mid
        if abs(val) < (deadzone / 127.5):
            return 0.0
        return round(val, 2)

    def on_left_stick_changed(self, stick: JoyStick):
        self.left_joystick = (
            self.normalize(stick.x),
            self.normalize(stick.y)
        )

    def on_right_stick_changed(self, stick: JoyStick):
        self.right_joystick = (
            self.normalize(stick.x),
            self.normalize(stick.y)
        )

    def on_left_trigger_changed(self, value: Number):
        # Increase only when trigger is pressed deeper
        if value > self.last_l2_value:
            delta = (value / 255.0) * 0.05
            self.gripper_value -= delta
            self.gripper_value = max(-1.0, self.gripper_value)
        self.last_l2_value = value

    def on_right_trigger_changed(self, value: Number):
        # Increase only when trigger is being pressed deeper
        if value > self.last_r2_value:
            delta = (value / 255.0) * 0.05
            self.gripper_value += delta
            self.gripper_value = min(1.0, self.gripper_value)
        self.last_r2_value = value

    def get_joystick_values(self):
        lx, ly = self.left_joystick
        rx, ry = self.right_joystick

        scaling_factor = 0.005
    
        return [-scaling_factor*ly, -scaling_factor*lx, -scaling_factor*ry, 10*scaling_factor*rx]

    # Print list of pressed buttons
    def get_pressed_buttons(self):
        return list(self.pressed_buttons)
    
    def get_gripper_value(self):
        return round(self.gripper_value, 2)

def main():
    driver = ps5_controller_driver()
    driver.controller.activate()
    try:
        while driver.is_running:
            #time.sleep(0.1)
            print("Joystick:", driver.get_joystick_values())
            print("Buttons:", driver.get_pressed_buttons())
            print("Gripper Value:", driver.get_gripper_value())

            if "cross" not in driver.get_pressed_buttons():
                print(type(driver.get_pressed_buttons()[0]))
    except KeyboardInterrupt:
        pass
    finally:
        driver.controller.deactivate()

if __name__ == "__main__":
    main()

