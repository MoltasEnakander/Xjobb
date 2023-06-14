import os

s = ["c1003", "c1052","c2154","c2236","c2560","c2588","c2663","c2746","c3014","c3066","c3074","c3090","c3241","c3308","c3364","c3376","c3643","c3689","c3698","c3720","c3789","c3983","c3987","c3998","c4043","c4076","c4083","c4164","c4168","c4287","c4215","c4228","c4242","c4289","c4405","c4424","c4442","c4475","c4495","c4496"]

for file in os.listdir("/data/largedataset/isars1"):
    if any(substring in file for substring in s):
        os.rename(os.path.join("/data/largedataset/isars1/", file), os.path.join("/data/largedataset/isars2/", file))

print(len(os.listdir("/data/largedataset/isars1")))
print(len(os.listdir("/data/largedataset/isars2")))

for file in os.listdir("/data/largedataset/rcs1"):
    if any(substring in file for substring in s):
        os.rename(os.path.join("/data/largedataset/rcs1/", file), os.path.join("/data/largedataset/rcs2/", file))

print(len(os.listdir("/data/largedataset/rcs1")))
print(len(os.listdir("/data/largedataset/rcs2")))

for file in os.listdir("/data/largedataset/scenes1"):
    if any(substring in file for substring in s):
        os.rename(os.path.join("/data/largedataset/scenes1/", file), os.path.join("/data/largedataset/scenes2/", file))

print(len(os.listdir("/data/largedataset/scenes1")))
print(len(os.listdir("/data/largedataset/scenes2")))
