# AI Game-Playing System: Project Structure and Core Types

## Directory Structure

```
ai-game-player/
├── game_interface/           # C++ game interaction components
│   ├── include/             # Public C++ headers
│   │   ├── frame_types.h    # Frame data structures
│   │   ├── input_types.h    # Input event structures
│   │   ├── capture.h        # Screen capture interface
│   │   └── simulation.h     # Input simulation interface
│   └── src/                 # C++ implementation files
│       ├── dxgi_capture.cpp # DirectX capture implementation
│       └── win_input.cpp    # Windows input implementation
├── ml_core/                 # Python ML components
│   ├── models/             # Neural network definitions
│   ├── training/           # Training pipeline
│   └── inference/          # Real-time inference
├── data_processing/         # Python data handling
│   ├── capture/            # Frame processing
│   ├── sync/               # Data synchronization
│   └── storage/            # Database operations
└── monitoring/             # Python monitoring tools
    ├── ui/                 # Control interface
    └── analytics/          # Performance tracking
```

## Core Data Structures

### Frame Data Types
```cpp
// game_interface/include/frame_types.h
#pragma once
#include <cstdint>
#include <vector>
#include <chrono>

namespace aigp {

struct FrameMetadata {
    uint64_t timestamp;      // Microseconds since epoch
    uint32_t width;          // Frame width in pixels
    uint32_t height;         // Frame height in pixels
    uint32_t channels;       // Number of color channels (typically 4 for RGBA)
    uint32_t sequence_num;   // Frame sequence number in session
    
    // Serialization methods
    std::vector<uint8_t> serialize() const;
    static FrameMetadata deserialize(const std::vector<uint8_t>& data);
};

struct Frame {
    FrameMetadata metadata;
    std::vector<uint8_t> data;  // Raw frame data in RGBA format
    
    // Frame operations
    bool isEmpty() const { return data.empty(); }
    size_t byteSize() const { return data.size(); }
    
    // Serialization methods
    std::vector<uint8_t> serialize() const;
    static Frame deserialize(const std::vector<uint8_t>& data);
};

} // namespace aigp
```

### Input Event Types
```cpp
// game_interface/include/input_types.h
#pragma once
#include <cstdint>
#include <variant>

namespace aigp {

enum class InputType {
    KEYBOARD,
    MOUSE_MOVE,
    MOUSE_BUTTON,
    GAMEPAD
};

struct KeyboardEvent {
    uint16_t virtual_key;
    bool is_pressed;
    bool is_extended;
    
    std::vector<uint8_t> serialize() const;
    static KeyboardEvent deserialize(const std::vector<uint8_t>& data);
};

struct MouseMoveEvent {
    int32_t x;
    int32_t y;
    int32_t delta_x;
    int32_t delta_y;
    
    std::vector<uint8_t> serialize() const;
    static MouseMoveEvent deserialize(const std::vector<uint8_t>& data);
};

struct MouseButtonEvent {
    uint8_t button;  // 0 = left, 1 = right, 2 = middle
    bool is_pressed;
    int32_t x;
    int32_t y;
    
    std::vector<uint8_t> serialize() const;
    static MouseButtonEvent deserialize(const std::vector<uint8_t>& data);
};

struct GamepadEvent {
    uint8_t button;
    float value;  // For analog inputs
    
    std::vector<uint8_t> serialize() const;
    static GamepadEvent deserialize(const std::vector<uint8_t>& data);
};

struct InputEvent {
    uint64_t timestamp;  // Microseconds since epoch
    InputType type;
    std::variant<KeyboardEvent, MouseMoveEvent, MouseButtonEvent, GamepadEvent> data;
    
    // Serialization methods
    std::vector<uint8_t> serialize() const;
    static InputEvent deserialize(const std::vector<uint8_t>& data);
};

} // namespace aigp
```

## Build Configuration

### CMake Configuration
```cmake
# CMakeLists.txt
cmake_minimum_required(VERSION 3.15)
project(ai-game-player)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Dependencies
find_package(DirectX REQUIRED)
find_package(ZeroMQ REQUIRED)

# Game interface library
add_library(game_interface
    src/dxgi_capture.cpp
    src/win_input.cpp
)

target_include_directories(game_interface
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
    PRIVATE
        ${DirectX_INCLUDE_DIRS}
        ${ZeroMQ_INCLUDE_DIRS}
)

target_link_libraries(game_interface
    PRIVATE
        ${DirectX_LIBRARIES}
        ${ZeroMQ_LIBRARIES}
)

# Install targets
install(TARGETS game_interface
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/
    DESTINATION include/aigp
)
```

## Python Package Configuration

### Package Setup
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name="ai-game-player",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "pyzmq>=22.0.0",
        "opencv-python>=4.5.0",
        "pandas>=1.3.0",
        "sqlalchemy>=1.4.0"
    ],
    python_requires=">=3.8",
)
```
