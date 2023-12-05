"""
This script turns a string into a list of key presses and presents
one character per second to help the user type them in a format 
that the model can process.
"""

import time
import os
import interpret

string = ("(@LazyDog_85's post) The quick, brown fox jumps over 13 lazy dogs at 7:03 PM; "
          "however, it's too #TIRED to continue!")
listified = interpret.listify_string(string)

clear = "cls" if os.name == "nt" else "clear"  # clear command for Windows or Linux

# print the listified string one character at a time
for i, ch in enumerate(listified):
    os.system(clear)
    print(f"[{listified[i]}] ", "  ".join(listified[i + 1: i + 20]))
    time.sleep(1)
