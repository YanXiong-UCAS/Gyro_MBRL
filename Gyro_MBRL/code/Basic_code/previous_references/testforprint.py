import sys
import os
class Logger(object):
    def __init__(self, filename = "Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('a.txt')

a = [1,2,3,4,5,6,7,8,9,0]

print(path)
print(os.path.dirname(__file__))
print("----------")



f = "a.txt"

a = 8
with open(f, "a") as file:
    for i in range(a):
        file.write(str(i) + "d" + " " + "\n")
    a += 1