import coremltools as ct
from nets import blaze_palm, blaze_palm_without_last_layer

# convert to Core ML and check predictions
model = blaze_palm_without_last_layer.build_blaze_palm_model()
mlmodel = ct.convert(model, convert_to="mlprogram")
