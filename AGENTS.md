# AGENTS.md

## Mission
Build and maintain a production-quality Godot GDExtension that integrates ONNX Runtime for local inference. The library must be safe in both the Godot editor and exported games, ergonomic from GDScript/C#, reproducible across desktop platforms, and designed so that CPU inference works first-class while optional GPU providers can be added without architectural churn.

## Project defaults
- Target runtime: Godot 4.6.x by default.
- Extension mechanism: GDExtension with `godot-cpp`.
- Primary supported platforms for the first production release: Windows x86_64, Linux x86_64/arm64, macOS arm64.
- Build workflow for this repository: CMake + Ninja + CMake Presets.
- Language baseline: C++20.
- Dependency policy: pin exact revisions/tags for every third-party dependency.
- ONNX Runtime default mode: CPU provider required; provider abstraction must allow optional CUDA, DirectML, CoreML, or other EPs later.

## Stability policy
- Prefer stable, pinned upstream revisions over floating branches.
- Do not silently follow `master`, `main`, or nightly packages in release builds.
- Do not raise the minimum Godot or compiler version without updating CI, docs, packaging, and migration notes in the same change.
- Keep the public Godot-facing API conservative and stable. Experimental features belong behind opt-in build flags or explicitly named experimental classes.

## Toolchain rules
Use a modern but conservative toolchain:
- CMake presets are the source of truth for local development and CI.
- Ninja is the default generator.
- Export `compile_commands.json`.
- Enable strict warnings on every compiler and treat warnings as errors in CI.
- Provide `Debug`, `RelWithDebInfo`, and `Release` presets.
- Enable ASan + UBSan on supported Clang/GCC debug builds. TSan is optional and advisory only.
- Enable LTO/IPO only in release presets where it is proven stable.
- Check formatting and linting in CI, not just locally.

Recommended repository files:
- `CMakeLists.txt`
- `CMakePresets.json`
- `cmake/`
- `vcpkg.json` if package-management is needed
- `.clang-format`
- `.clang-tidy`
- `.editorconfig`
- `.github/workflows/`
- `tests/`
- `example/` or `project/` Godot demo project
- `docs/`

## Build-system stance
Even though `godot-cpp` is historically SCons-first, this repository is CMake-first because ONNX Runtime integration, reproducible dependency wiring, IDE support, and CI orchestration are materially cleaner in CMake.

Rules:
- Keep the CMake path healthy at all times.
- Do not add local build instructions that only work through an IDE UI.
- If a helper script is added, it must delegate to CMake presets rather than re-encode build logic.
- If upstream `godot-cpp` behavior forces a temporary SCons workaround, isolate it behind a script or documented compatibility layer rather than leaking it throughout the project.

## Dependency management
### `godot-cpp`
- Pin an exact branch/tag/commit compatible with the minimum supported Godot version.
- Prefer a vendored submodule or another explicit, reproducible pin.
- Do not assume ABI compatibility across unrelated Godot minor versions.
- Generate the `.gdextension` manifest from canonical build metadata, not hand-maintained duplicated strings where avoidable.

### ONNX Runtime
- Prefer official release artifacts for day-to-day development.
- Build ONNX Runtime from source only when a released package does not satisfy one of these needs:
  - required execution provider combination
  - binary size reduction / minimal build
  - platform-specific patching
  - feature only available from source configuration
- If building ORT from source, pin an official release branch/tag and document the exact configure/build flags.
- If custom ORT builds are used, capture checksums, license notices, and the build recipe in-repo.
- Do not rebuild ONNX Runtime from source as part of the default incremental developer build.

### Other dependencies
- Keep the dependency graph minimal.
- Prefer standard library facilities first.
- Any new dependency must have a concrete justification in the PR description: what it replaces, why the standard library is insufficient, and how it affects packaging size and licenses.

## Architecture rules
Separate the project into three layers.

### 1. Godot boundary layer
Responsible for:
- class registration
- `Object`/`RefCounted`/`Resource` integration
- `Variant` conversion
- signals, properties, editor-facing API
- marshaling results back to the main thread

This layer must be thin. It must not contain ONNX Runtime policy, tensor math, or ad-hoc concurrency logic.

### 2. Core inference layer
Responsible for:
- session creation and caching
- provider selection and session options
- model metadata inspection
- input/output validation
- tensor conversion between internal native representations and ORT values
- synchronous inference
- asynchronous job execution without direct SceneTree interaction

This layer should be regular C++ with minimal dependency on Godot headers.

### 3. Platform and packaging layer
Responsible for:
- shared library naming
- runtime loading details
- provider-specific packaging
- post-build copy/layout steps
- release archives and notices

Keep platform-specific code localized.

## Public API shape
The Godot-facing API should be small and unsurprising. Favor a narrow MVP that is easy to stabilize.

Minimum capability set:
- load a model from disk
- inspect model inputs and outputs
- run synchronous inference on CPU
- run asynchronous inference without blocking the main thread
- return structured errors and metadata
- expose provider and optimization settings in a controlled way

Preferred Godot-facing objects:
- `OrtSession` or similarly named `RefCounted` wrapper for model/session lifecycle
- `OrtTensorSpec` / `OrtModelMetadata` style objects for shape/type metadata
- `OrtProviderConfig` style resource or value object for provider settings

Do not expose raw `Ort::*` types to scripts.
Do not require script authors to reason about allocators, raw pointers, or provider-specific C APIs.

## C++ design rules
Use modern C++ deliberately, not performatively.

Required style:
- RAII for every owned resource
- value semantics by default
- `std::unique_ptr` for exclusive ownership
- `std::shared_ptr` only with explicit ownership reasoning
- `std::span`, `std::string_view`, `std::optional`, `std::variant`, `std::array`, `std::filesystem` where appropriate
- `enum class` over plain enums
- `[[nodiscard]]` for meaningful status-returning functions
- `constexpr` for compile-time constants
- move operations and `noexcept` where they improve correctness/perf
- concepts only where they clearly simplify template constraints

Avoid:
- unnecessary template metaprogramming
- inheritance-heavy utility hierarchies
- macro-based abstractions except where required by Godot binding macros or platform/export definitions
- raw owning pointers
- hidden singleton state except for a deliberate process-wide ORT environment manager
- overuse of ranges when a plain loop is clearer
- C++ modules in the initial codebase
- coroutines unless there is a measured, architecture-level reason to adopt them

## Error handling rules
- No exception may cross the GDExtension boundary.
- If exceptions are enabled internally, catch them at the boundary and convert them to explicit Godot errors/messages.
- Prefer explicit result/status types for recoverable failures in core logic.
- Every failure path must include actionable context: model path, provider, input name, expected shape/type, actual shape/type, etc.
- Validation errors must be distinguished from internal/runtime failures.
- Never swallow ONNX Runtime error context.

## Concurrency rules
- Do not interact with the active Godot SceneTree from worker threads.
- Worker threads may run model loading, validation, tensor preprocessing, and ORT execution, but they must publish results back through a thread-safe queue and a main-thread handoff.
- Treat Godot containers as non-thread-safe unless externally synchronized.
- Prefer STL containers in core worker-thread code and convert to Godot containers at the boundary.
- Use one process-wide `Ort::Env`.
- Reuse `Ort::Session` objects; do not create a new session per inference call.
- If multiple sessions are expected in one process, design for shared/global threadpools and shared allocators where beneficial.
- Async APIs must be cancel-safe or at minimum shutdown-safe.
- Destruction order matters: background work must stop before Godot objects or ORT state is torn down.

## Performance rules
- Optimize for predictable latency and startup behavior before micro-optimizing throughput.
- Cache model metadata after load.
- Avoid repeated string conversions and repeated shape parsing.
- Separate session creation from hot inference paths.
- For non-CPU providers, design an I/O binding path to avoid needless host/device copies.
- If startup time is material, support optional offline-optimized model artifacts or serialized optimized models.
- If packaging size becomes a blocker, support reduced-operator ORT builds as an advanced build profile rather than the default path.

## Godot integration rules
- Register only the classes that are part of the supported API surface.
- Keep script-facing names concise and idiomatic for Godot users.
- Prefer `RefCounted` or `Resource` for long-lived data holders over forcing users to place helper nodes in the scene tree.
- Use signals for async completion and failure, not polling-only APIs.
- Keep editor-time behavior safe. Failed model loads in the editor must degrade gracefully instead of destabilizing the editor.
- Do not perform expensive work in constructors; use explicit `load_*`/`initialize_*` steps.

## Type-conversion rules
Build conversion code as a dedicated subsystem.

Requirements:
- Centralize mappings between Godot `Variant`/typed arrays and ORT tensor element types.
- Validate rank, shape, contiguous layout assumptions, and dtype before constructing `Ort::Value`.
- Keep conversion code test-heavy.
- Do not duplicate conversion logic across multiple Godot classes.
- Handle unsupported types explicitly with high-quality diagnostics.

Initial MVP support should prioritize:
- `float32`
- `int64`
- `bool`
- plain dense tensors

Advanced types such as strings, half-precision, images-as-tensors, or sequence/map outputs should be added intentionally and documented individually.

## Testing strategy
Three test layers are required.

### Native unit tests
Test pure C++ logic without Godot runtime dependence where possible:
- provider parsing
- session options construction
- shape validation
- type conversion helpers
- result/error mapping
- cache keys and session reuse logic

### Native integration tests
Use small deterministic ONNX fixtures to verify:
- model load
- metadata extraction
- successful inference
- failure behavior on bad input shapes/types
- multiple session behavior

### Godot integration smoke tests
Use a sample Godot project to verify:
- extension loads in editor/runtime
- GDScript can call the API
- async signal flow works
- packaged binaries are found by the `.gdextension` file

Rules:
- Tests must be deterministic.
- Do not rely on network access.
- Do not require GPU hardware in the default PR pipeline.
- GPU/provider-specific tests belong in optional workflows or dedicated runners.

## CI rules
Required CI jobs:
- format/lint
- Linux build + tests
- Windows build + tests
- macOS build + at least build + smoke checks
- package artifact assembly

Preferred extras:
- sanitizer job on Linux/macOS
- release workflow producing zipped Godot addon artifacts
- nightly compatibility matrix for alternate Godot/ORT/provider combinations

CI must fail on:
- formatting drift
- lint errors
- warnings treated as errors
- test failures
- missing packaged files

## Packaging and release rules
Release artifacts must include:
- compiled extension binaries
- `.gdextension` manifest
- sample/demo project or minimal usage example
- README and API notes
- third-party license notices
- version metadata

Rules:
- Do not ship ad-hoc local paths in manifests.
- Keep runtime library layout deterministic.
- Verify the packaged artifact in a clean Godot project before release.
- Version the Godot addon/library independently from upstream dependency versions.

## Documentation rules
Every feature PR must update the relevant docs.

Minimum docs set:
- build instructions
- supported platforms/providers
- supported tensor types
- sync and async API examples
- packaging notes
- known limitations
- troubleshooting section

Every public class/function exposed to Godot should have at least one example in either docs or the example project.

## Security and trust boundaries
- Do not auto-download models at runtime.
- Do not auto-load arbitrary provider plugins or custom op libraries from untrusted paths.
- Treat model files as untrusted input: validate paths, validate load failures, and do not assume shape/type sanity.
- Keep file-system access explicit and documented.
- If custom ops are ever supported, ship them as an explicit advanced feature with separate documentation and threat notes.

## Definition of done for code changes
A change is done only when all of the following are true:
- builds from a clean checkout using documented commands
- passes formatting, lint, and tests
- preserves or improves cross-platform behavior
- updates docs and example usage if the public API changed
- includes packaging changes when dependency/runtime layout changed
- includes regression tests for every bug fix
- does not introduce hidden global state or thread-lifetime hazards

## Priority order when making trade-offs
1. Correctness and editor/runtime safety
2. Stable API and maintainability
3. Reproducible builds and packaging
4. Cross-platform portability
5. Performance and binary-size tuning
6. Additional providers/features

## Anti-patterns to reject in review
Reject or refactor changes that:
- create an ORT environment/session on every inference call
- manipulate SceneTree objects from worker threads
- expose raw ONNX Runtime handles directly to scripts
- hardcode machine-local paths
- add dependencies without a packaging and license plan
- duplicate tensor conversion code
- rely on undefined ownership or manual lifetime juggling
- mix Godot UI/editor logic into the core inference layer
- optimize for a GPU provider before the CPU path is robust
