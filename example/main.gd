extends Node
## Demonstrates gonx async model load plus sync/async inference.

var _session: OrtSession
var _inputs := {
	"A": PackedFloat32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
	"B": PackedFloat32Array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]),
}


func _ready() -> void:
	print("=== gonx Example ===")
	print("Available providers: ", OrtProviderConfig.get_available_providers())

	_session = OrtSession.new()
	_session.model_load_started.connect(_on_model_load_started)
	_session.model_loaded.connect(_on_model_loaded)
	_session.model_load_failed.connect(_on_model_load_failed)
	_session.model_load_cancelled.connect(_on_model_load_cancelled)
	_session.inference_completed.connect(_on_inference_completed)
	_session.inference_failed.connect(_on_inference_failed)
	_session.inference_cancelled.connect(_on_inference_cancelled)

	var config := OrtProviderConfig.new()
	config.provider = "CPU"
	config.intra_op_threads = 2
	config.optimization_level = 99

	var model_path := "res://models/add_floats.onnx"
	var request_id := _session.load_model_with_config_async(model_path, config)
	if request_id == 0:
		push_error("Failed to start async load: %s" % _session.get_last_error())
		get_tree().quit()


func _on_model_load_started(request_id: int, path: String) -> void:
	print("Loading model asynchronously: ", request_id, " -> ", path)


func _on_model_loaded(request_id: int, path: String) -> void:
	print("Model loaded: ", request_id, " -> ", path)
	print("Resolved model path: ", _session.get_model_path())

	print("\nInputs:")
	for spec: OrtTensorSpec in _session.get_input_specs():
		print("  ", spec.describe())

	print("\nOutputs:")
	for spec: OrtTensorSpec in _session.get_output_specs():
		print("  ", spec.describe())

	var meta := _session.get_metadata()
	print("\nMetadata:")
	print("  Producer: ", meta.producer_name)
	print("  Graph: ", meta.graph_name)

	print("\n--- Sync Inference ---")
	var sync_result := _session.run_inference(_inputs)
	if sync_result.is_empty():
		push_error("Inference failed: %s" % _session.get_last_error())
		get_tree().quit()
		return

	print("Output C: ", sync_result.get("C", []))
	print("Output C shape: ", sync_result.get("C_shape", []))

	print("\n--- Async Inference ---")
	var inference_request_id := _session.run_inference_async(_inputs)
	if inference_request_id == 0:
		push_error("Async inference failed to start: %s" % _session.get_last_error())
		get_tree().quit()


func _on_model_load_failed(request_id: int, error_code: int, error: String) -> void:
	push_error("Async model load failed (%s / %s): %s" % [request_id, error_code, error])
	get_tree().quit()


func _on_model_load_cancelled(request_id: int) -> void:
	push_warning("Async model load cancelled: %s" % request_id)
	get_tree().quit()


func _on_inference_completed(request_id: int, result: Dictionary) -> void:
	print("Async result (%s): " % request_id, result.get("C", []))
	print("Done!")
	get_tree().quit()


func _on_inference_failed(request_id: int, error_code: int, error: String) -> void:
	push_error("Async inference failed (%s / %s): %s" % [request_id, error_code, error])
	get_tree().quit()


func _on_inference_cancelled(request_id: int) -> void:
	push_warning("Async inference cancelled: %s" % request_id)
	get_tree().quit()
