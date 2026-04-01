extends Node
## Demonstrates basic gonx usage: load model, inspect, run sync and async inference.

func _ready() -> void:
	print("=== gonx Example ===")

	# --- Provider info ---
	var providers := OrtProviderConfig.get_available_providers()
	print("Available providers: ", providers)

	# --- Create and configure session ---
	var session := OrtSession.new()

	# Optional: configure provider settings
	var config := OrtProviderConfig.new()
	config.provider = "CPU"
	config.intra_op_threads = 2
	config.optimization_level = 99  # ORT_ENABLE_ALL

	# --- Load a model ---
	# Place an ONNX model at res://models/add_floats.onnx
	# (you can generate it with tests/fixtures/generate_test_models.py)
	var model_path := "res://models/add_floats.onnx"
	var err := session.load_model_with_config(model_path, config)
	if err != 0:
		push_error("Failed to load model: %s" % session.get_last_error())
		return

	print("Model loaded: ", session.get_model_path())

	# --- Inspect inputs/outputs ---
	print("\nInputs:")
	for spec: OrtTensorSpec in session.get_input_specs():
		print("  ", spec.describe())

	print("\nOutputs:")
	for spec: OrtTensorSpec in session.get_output_specs():
		print("  ", spec.describe())

	# --- Inspect metadata ---
	var meta := session.get_metadata()
	print("\nMetadata:")
	print("  Producer: ", meta.producer_name)
	print("  Graph: ", meta.graph_name)

	# --- Synchronous inference ---
	print("\n--- Sync Inference ---")
	var inputs := {
		"A": PackedFloat32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
		"B": PackedFloat32Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
	}

	var result := session.run_inference(inputs)
	if result.is_empty():
		push_error("Inference failed: %s" % session.get_last_error())
	else:
		print("Output C: ", result.get("C", []))
		print("Output C shape: ", result.get("C_shape", []))

	# --- Async inference ---
	print("\n--- Async Inference ---")
	session.inference_completed.connect(_on_inference_completed)
	session.inference_failed.connect(_on_inference_failed)
	session.run_inference_async(inputs)


func _on_inference_completed(result: Dictionary) -> void:
	print("Async result: ", result.get("C", []))
	print("Done!")
	get_tree().quit()


func _on_inference_failed(error: String) -> void:
	push_error("Async inference failed: %s" % error)
	get_tree().quit()
