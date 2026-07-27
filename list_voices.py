from kokoro_onnx import Kokoro
kokoro = Kokoro("data/models/kokoro-v1.0.onnx", "data/models/voices-v1.0.bin")
print(kokoro.get_voices())
