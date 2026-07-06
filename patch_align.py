import re

with open('public/js/swanMath.js', 'r') as f:
    code = f.read()

# Pad MatMul config
code = code.replace(
    "const configBuf = tempBuf(new Uint32Array([inDim, outDim, seqLen]));",
    "const configBuf = tempBuf(new Uint32Array([inDim, outDim, seqLen, 0]));"
)

# Pad addConfig, siluConfig, embedConfig
code = code.replace(
    "tempBuf(new Uint32Array([this.DIM, seqLen])",
    "tempBuf(new Uint32Array([this.DIM, seqLen, 0, 0])"
)
code = code.replace(
    "tempBuf(new Uint32Array([8960, seqLen])",
    "tempBuf(new Uint32Array([8960, seqLen, 0, 0])"
)
code = code.replace(
    "rmsnormConfigData = new ArrayBuffer(12);",
    "rmsnormConfigData = new ArrayBuffer(16);"
)

code = code.replace(
    "tempBuf(new Uint32Array([12, 2, 128, startPos, seqLen]),",
    "tempBuf(new Uint32Array([12, 2, 128, startPos, seqLen, 0, 0, 0]),"
)

with open('public/js/swanMath.js', 'w') as f:
    f.write(code)

print("Padded uniforms")
