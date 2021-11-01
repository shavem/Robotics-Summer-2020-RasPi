import smbus
import math
import time
import os
import pexpect
import sys


# 1 - right
# 2 - left

# Register
power_mgmt_1 = 0x6b
power_mgmt_2 = 0x6c

with open("/home/pi/Documents/password.txt") as f:
    pas = f.readline()


def read_byte(reg):
    return bus.read_byte_data(address, reg)


def read_word(reg):
    h = bus.read_byte_data(address, reg)
    l = bus.read_byte_data(address, reg + 1)
    value = (h << 8) + l
    return value


def read_word_2c(reg):
    val = read_word(reg)
    if (val >= 0x8000):
        return -((65535 - val) + 1)
    else:
        return val


def dist(a, b):
    return math.sqrt((a * a) + (b * b))


def get_y_rotation(x, y, z):
    radians = math.atan2(x, dist(y, z))
    return -math.degrees(radians)


def get_x_rotation(x, y, z):
    radians = math.atan2(y, dist(x, z))
    return math.degrees(radians)


bus = smbus.SMBus(1)  # bus = smbus.SMBus(0)
address = 0x68  # via i2cdetect

cycle = 0
while cycle < 10:
    label = raw_input("activity label: ")
    f = open(f"data{cycle}.csv", "w")
    for i in range(1000):
        # Activate to be able to address the module
        bus.write_byte_data(address, power_mgmt_1, 0)

        acceleration_xout = read_word_2c(0x3b)
        acceleration_yout = read_word_2c(0x3d)
        acceleration_zout = read_word_2c(0x3f)

        acceleration_xout_scaled = acceleration_xout / 16384.0
        acceleration_yout_scaled = acceleration_yout / 16384.0
        acceleration_zout_scaled = acceleration_zout / 16384.0

        # print "acceleration_xout: ", ("%6d" % acceleration_xout), " Scaled: ", acceleration_xout_scaled
        # print "acceleration_yout: ", ("%6d" % acceleration_yout), " Scaled: ", acceleration_yout_scaled
        # print "acceleration_zout: ", ("%6d" % acceleration_zout), " Scaled: ", acceleration_zout_scaled

        # x_rotation = get_x_rotation(acceleration_xout_scaled, acceleration_yout_scaled, acceleration_zout_scaled)
        # y_rotation = get_y_rotation(acceleration_xout_scaled, acceleration_yout_scaled, acceleration_zout_scaled)

        # print "X Rotation: " , x_rotation
        # print "Y Rotation: " , y_rotation
        x = '{0},{1},{2},{3},{4} \n'.format(i, acceleration_xout, acceleration_yout, acceleration_zout, int(label))
        f.write(x)

        time.sleep(0.0001)

    f.close()
    print(f"data{cycle}.txt printed")
    cycle += 1
os.system("git add -A")
os.system("git commit -m data")
child = pexpect.spawn("git push")
child.logfile_read = sys.stdout
child.expect("Username for 'https://github.com':")
child.sendline("Shavem")
child.expect("Password for 'https://Shavem@github.com':")
child.sendline(pas)
child.interact()