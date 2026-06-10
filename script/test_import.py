import sys
print('cwd in path:', '' in sys.path)
import ultralytics
print('ultralytics file:', ultralytics.__file__)
print('nn file:', ultralytics.nn.__file__)
# Check if yolo subpackage exists
import ultralytics.yolo
print('yolo file:', ultralytics.yolo.__file__)
