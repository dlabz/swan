import re

with open('public/js/swanMath.js', 'r') as f:
    code = f.read()

# I want to restore it to the proper form without destroy
code = code.replace("for (const buf of transientBuffers) { // buf.destroy(); }", "")
code = code.replace("for (const buf of transientBuffers) { // buf.destroy();\n", "")

with open('public/js/swanMath.js', 'w') as f:
    f.write(code)

print("Syntax fixed")
