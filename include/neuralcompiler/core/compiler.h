#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include "graph.h"
#include "tensor.h"

namespace neuralcompiler {

enum class OptimizationLevel {
    O0,  // No optimization
    O1,  // Basic optimizations
    O2,  // Standard optimizations
    O3   // Aggressive optimizations
};

enum class Backend {
    CPU,
    CUDA,
    ACCELERATOR
};

struct CompilerConfig {
    OptimizationLevel optimization_level = OptimizationLevel::O2;
    Backend target_backend = Backend::ACCELERATOR;
    bool enable_quantization = false;
    bool enable_fusion = true;
    bool enable_memory_optimization = true;
    int num_threads = 8;
    size_t memory_limit_mb = 4096;
};

class OptimizationPass;
class CodegenBackend;

class Compiler {
public:
    explicit Compiler(const CompilerConfig& config);
    ~Compiler();

    // Main compilation interface
    std::unique_ptr<CompiledModel> compile(const Graph& input_graph);
    
    // Register custom optimization passes
    void registerOptimizationPass(std::unique_ptr<OptimizationPass> pass);
    
    // Performance analysis
    CompilationStats getLastCompilationStats() const;

private:
    CompilerConfig config_;
    std::vector<std::unique_ptr<OptimizationPass>> optimization_passes_;
    std::unique_ptr<CodegenBackend> backend_;
    
    // Internal compilation phases
    Graph runFrontendPasses(const Graph& graph);
    Graph runOptimizationPasses(const Graph& graph);
    std::unique_ptr<CompiledModel> runCodegenPass(const Graph& graph);
    
    void initializeOptimizationPasses();
    void initializeBackend();
};

// Compiled model container
class CompiledModel {
public:
    CompiledModel(std::unique_ptr<CodegenBackend> backend, const Graph& optimized_graph);
    
    // Code generation
    std::string generateCode() const;
    std::vector<uint8_t> generateBinary() const;
    
    // Runtime interface
    void setInput(const std::string& name, const Tensor& tensor);
    Tensor getOutput(const std::string& name) const;
    void execute();
    
    // Performance metrics
    double getInferenceTime() const;
    size_t getMemoryUsage() const;

private:
    std::unique_ptr<CodegenBackend> backend_;
    Graph optimized_graph_;
    std::unordered_map<std::string, Tensor> inputs_;
    std::unordered_map<std::string, Tensor> outputs_;
};

struct CompilationStats {
    double total_time_ms;
    double frontend_time_ms;
    double optimization_time_ms;
    double codegen_time_ms;
    size_t original_ops_count;
    size_t optimized_ops_count;
    size_t memory_saved_bytes;
};

} // namespace neuralcompiler

// ============================================================================
// File: include/neuralcompiler/optimization/fusion_pass.h
// ============================================================================
#pragma once

#include "../core/graph.h"
#include "optimization_pass.h"
#include <vector>

namespace neuralcompiler {

class FusionPass : public OptimizationPass {
public:
    FusionPass();
    ~FusionPass() override = default;

    bool runOnGraph(Graph& graph) override;
    std::string getPassName() const override { return "FusionPass"; }

    // CNN-specific fusion patterns
    bool fuseConvBatchNormReLU(Graph& graph);
    bool fuseDepthwiseConv(Graph& graph);
    bool fuseConvAdd(Graph& graph);
    
    // Transformer-specific fusion patterns
    bool fuseMultiHeadAttention(Graph& graph);
    bool fuseFeedForward(Graph& graph);
    bool fuseLayerNorm(Graph& graph);
    
    // General fusion patterns
    bool fuseElementwiseOps(Graph& graph);
    bool fuseMatMulAdd(Graph& graph);

private:
    struct FusionPattern {
        std::vector<NodeType> pattern;
        NodeType fused_type;
        std::function<Node(const std::vector<Node>&)> fusion_func;
    };
    
    std::vector<FusionPattern> cnn_patterns_;
    std::vector<FusionPattern> transformer_patterns_;
    
    bool matchPattern(const Graph& graph, const FusionPattern& pattern, 
                     size_t start_node, std::vector<size_t>& matched_nodes);
    Node createFusedNode(const std::vector<Node>& nodes, const FusionPattern& pattern);
    bool isValidFusion(const std::vector<Node>& nodes) const;
    
    // Pattern matching utilities
    bool isConvBNReluPattern(const Graph& graph, size_t conv_idx) const;
    bool isAttentionPattern(const Graph& graph, size_t start_idx) const;
    
    size_t nodes_fused_ = 0;
};

} // namespace neuralcompiler

// ============================================================================
// File: src/optimization/fusion_pass.cpp
// ============================================================================
#include "neuralcompiler/optimization/fusion_pass.h"
#include "neuralcompiler/core/node.h"
#include <algorithm>
#include <unordered_set>

namespace neuralcompiler {

FusionPass::FusionPass() {
    // Initialize CNN fusion patterns
    cnn_patterns_.push_back({
        {NodeType::CONV2D, NodeType::BATCH_NORM, NodeType::RELU},
        NodeType::FUSED_CONV_BN_RELU,
        [](const std::vector<Node>& nodes) -> Node {
            Node fused_node(NodeType::FUSED_CONV_BN_RELU, "fused_conv_bn_relu");
            // Combine attributes from all nodes
            fused_node.copyAttributesFrom(nodes[0]); // Conv attributes
            fused_node.addAttributesFrom(nodes[1]);  // BatchNorm attributes
            return fused_node;
        }
    });
    
    // Initialize transformer fusion patterns
    transformer_patterns_.push_back({
        {NodeType::MATMUL, NodeType::ADD, NodeType::SOFTMAX},
        NodeType::FUSED_ATTENTION,
        [](const std::vector<Node>& nodes) -> Node {
            Node fused_node(NodeType::FUSED_ATTENTION, "fused_attention");
            fused_node.copyAttributesFrom(nodes[0]);
            return fused_node;
        }
    });
}

bool FusionPass::runOnGraph(Graph& graph) {
    bool modified = false;
    nodes_fused_ = 0;
    
    // Apply CNN-specific fusions
    modified |= fuseConvBatchNormReLU(graph);
    modified |= fuseDepthwiseConv(graph);
    modified |= fuseConvAdd(graph);
    
    // Apply transformer-specific fusions
    modified |= fuseMultiHeadAttention(graph);
    modified |= fuseFeedForward(graph);
    
    // Apply general fusions
    modified |= fuseElementwiseOps(graph);
    modified |= fuseMatMulAdd(graph);
    
    if (modified) {
        graph.topologicalSort();
        graph.validateGraph();
    }
    
    return modified;
}

bool FusionPass::fuseConvBatchNormReLU(Graph& graph) {
    bool modified = false;
    auto nodes = graph.getNodes();
    
    for (size_t i = 0; i < nodes.size() - 2; ++i) {
        if (isConvBNReluPattern(graph, i)) {
            // Create fused node
            Node fused_node(NodeType::FUSED_CONV_BN_RELU, 
                           "fused_conv_bn_relu_" + std::to_string(i));
            
            // Copy and merge attributes from conv, batchnorm, relu
            const auto& conv_node = nodes[i];
            const auto& bn_node = nodes[i + 1];
            const auto& relu_node = nodes[i + 2];
            
            fused_node.copyAttributesFrom(conv_node);
            fused_node.addAttributesFrom(bn_node);
            
            // Set inputs and outputs
            fused_node.setInputs(conv_node.getInputs());
            fused_node.setOutputs(relu_node.getOutputs());
            
            // Replace nodes in graph
            graph.replaceNodeSequence(i, i + 2, fused_node);
            
            modified = true;
            nodes_fused_ += 3;
            
            // Adjust index since we removed nodes
            i = std::max(0, static_cast<int>(i) - 1);
        }
    }
    
    return modified;
}

bool FusionPass::fuseMultiHeadAttention(Graph& graph) {
    bool modified = false;
    auto nodes = graph.getNodes();
    
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (isAttentionPattern(graph, i)) {
            // Complex attention fusion logic
            std::vector<size_t> attention_nodes;
            
            // Find Q, K, V computation nodes
            for (size_t j = i; j < std::min(i + 10, nodes.size()); ++j) {
                if (nodes[j].getType() == NodeType::MATMUL ||
                    nodes[j].getType() == NodeType::SOFTMAX ||
                    nodes[j].getType() == NodeType::ADD) {
                    attention_nodes.push_back(j);
                }
            }
            
            if (attention_nodes.size() >= 4) { // Min nodes for attention
                Node fused_attention(NodeType::FUSED_ATTENTION, "fused_mha");
                
                // Set up fused attention attributes
                fused_attention.setAttribute("num_heads", 8);
                fused_attention.setAttribute("head_dim", 64);
                fused_attention.setAttribute("dropout", 0.1f);
                
                // Replace attention subgraph
                graph.replaceNodeSubgraph(attention_nodes, fused_attention);
                
                modified = true;
                nodes_fused_ += attention_nodes.size();
            }
        }
    }
    
    return modified;
}

bool FusionPass::fuseElementwiseOps(Graph& graph) {
    bool modified = false;
    auto nodes = graph.getNodes();
    
    // Look for chains of elementwise operations
    for (size_t i = 0; i < nodes.size() - 1; ++i) {
        if (nodes[i].isElementwise() && nodes[i + 1].isElementwise()) {
            // Can fuse consecutive elementwise operations
            std::vector<size_t> elementwise_chain = {i};
            
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                if (nodes[j].isElementwise() && 
                    graph.areConnected(elementwise_chain.back(), j)) {
                    elementwise_chain.push_back(j);
                } else {
                    break;
                }
            }
            
            if (elementwise_chain.size() >= 2) {
                Node fused_elementwise(NodeType::FUSED_ELEMENTWISE, "fused_eltwise");
                
                // Store original operations in fused node
                std::vector<NodeType> ops;
                for (size_t idx : elementwise_chain) {
                    ops.push_back(nodes[idx].getType());
                }
                fused_elementwise.setAttribute("fused_ops", ops);
                
                graph.replaceNodeSequence(elementwise_chain.front(), 
                                        elementwise_chain.back(), 
                                        fused_elementwise);
                
                modified = true;
                nodes_fused_ += elementwise_chain.size();
                
                i = elementwise_chain.back();
            }
        }
    }
    
    return modified;
}

bool FusionPass::isConvBNReluPattern(const Graph& graph, size_t conv_idx) const {
    const auto& nodes = graph.getNodes();
    
    if (conv_idx + 2 >= nodes.size()) return false;
    
    return nodes[conv_idx].getType() == NodeType::CONV2D &&
           nodes[conv_idx + 1].getType() == NodeType::BATCH_NORM &&
           nodes[conv_idx + 2].getType() == NodeType::RELU &&
           graph.areConnected(conv_idx, conv_idx + 1) &&
           graph.areConnected(conv_idx + 1, conv_idx + 2);
}

bool FusionPass::isAttentionPattern(const Graph& graph, size_t start_idx) const {
    // Look for pattern: MatMul -> Add -> Softmax -> MatMul (simplified)
    const auto& nodes = graph.getNodes();
    
    if (start_idx + 3 >= nodes.size()) return false;
    
    return nodes[start_idx].getType() == NodeType::MATMUL &&
           nodes[start_idx + 1].getType() == NodeType::ADD &&
           nodes[start_idx + 2].getType() == NodeType::SOFTMAX &&
           nodes[start_idx + 3].getType() == NodeType::MATMUL;
}

} // namespace neuralcompiler

// ============================================================================
// File: examples/cnn_optimization/resnet50_example.cpp
// ============================================================================
#include "neuralcompiler/compiler.h"
#include "neuralcompiler/frontend/onnx_parser.h"
#include "neuralcompiler/optimization/fusion_pass.h"
#include "neuralcompiler/optimization/memory_pass.h"

#include <iostream>
#include <chrono>

using namespace neuralcompiler;

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: " << argv[0] << " <resnet50.onnx>" << std::endl;
        return 1;
    }
    
    try {
        std::cout << "=== NeuralCompiler ResNet-50 Optimization Example ===" << std::endl;
        
        // Parse ONNX model
        std::cout << "Loading ResNet-50 model from: " << argv[1] << std::endl;
        ONNXParser parser;
        auto graph = parser.parseFromFile(argv[1]);
        
        std::cout << "Original model stats:" << std::endl;
        std::cout << "  - Nodes: " << graph.getNodeCount() << std::endl;
        std::cout << "  - Parameters: " << graph.getParameterCount() << std::endl;
        std::cout << "  - Memory: " << graph.getMemoryUsage() / (1024*1024) << " MB" << std::endl;
        
        // Configure compiler for aggressive CNN optimization
        CompilerConfig config;
        config.optimization_level = OptimizationLevel::O3;
        config.target_backend = Backend::ACCELERATOR;
        config.enable_quantization = true;
        config.enable_fusion = true;
        config.enable_memory_optimization = true;
        
        Compiler compiler(config);
        
        // Add custom CNN-specific optimizations
        auto fusion_pass = std::make_unique<FusionPass>();
        auto memory_pass = std::make_unique<MemoryPass>();
        
        compiler.registerOptimizationPass(std::move(fusion_pass));
        compiler.registerOptimizationPass(std::move(memory_pass));
        
        // Compile the model
        std::cout << "\nCompiling model..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        auto compiled_model = compiler.compile(graph);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto compile_time = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        // Get compilation statistics
        auto stats = compiler.getLastCompilationStats();
        
        std::cout << "\nCompilation Results:" << std::endl;
        std::cout << "  - Compilation time: " << compile_time << " ms" << std::endl;
        std::cout << "  - Optimization time: " << stats.optimization_time_ms << " ms" << std::endl;
        std::cout << "  - Nodes reduced: " << stats.original_ops_count 
                  << " -> " << stats.optimized_ops_count << std::endl;
        std::cout << "  - Memory saved: " << stats.memory_saved_bytes / (1024*1024) 
                  << " MB" << std::endl;
        
        // Generate optimized code
        std::cout << "\nGenerating accelerator code..." << std::endl;
        auto generated_code = compiled_model->generateCode();
        
        // Save generated code
        std::ofstream code_file("resnet50_optimized.cc");
        code_file << generated_code;
        code_file.close();
        
        std::cout << "Generated code saved to: resnet50_optimized.cc" << std::endl;
        
        // Performance test with dummy input
        std::cout << "\nRunning performance test..." << std::endl;
        
        // Create dummy input tensor (1, 3, 224, 224)
        Tensor input_tensor({1, 3, 224, 224}, DataType::FLOAT32);
        input_tensor.fillRandom();
        
        compiled_model->setInput("input", input_tensor);
        
        // Warm-up runs
        for (int i = 0; i < 5; ++i) {
            compiled_model->execute();
        }
        
        // Timed runs
        const int num_runs = 100;
        start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_runs; ++i) {
            compiled_model->execute();
        }
        
        end_time = std::chrono::high_resolution_clock::now();
        auto inference_time = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count() / num_runs;
        
        std::cout << "Performance Results:" << std::endl;
        std::cout << "  - Average inference time: " << inference_time / 1000.0 << " ms" << std::endl;
        std::cout << "  - Memory usage: " << compiled_model->getMemoryUsage() / (1024*1024) 
                  << " MB" << std::endl;
        
        // Get output tensor
        auto output = compiled_model->getOutput("output");
        std::cout << "  - Output shape: " << output.getShapeString() << std::endl;
        
        std::cout << "\n=== Optimization Complete ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

// ============================================================================
// File: CMakeLists.txt
// ============================================================================
cmake_minimum_required(VERSION 3.15)
project(NeuralCompiler VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options
option(ENABLE_CUDA "Enable CUDA backend" OFF)
option(ENABLE_TESTING "Enable testing" ON)
option(ENABLE_BENCHMARKS "Enable benchmarks" ON)
option(ENABLE_SANITIZERS "Enable sanitizers" OFF)

# Find required packages
find_package(Protobuf REQUIRED)
find_package(Threads REQUIRED)

if(ENABLE_CUDA)
    find_package(CUDA REQUIRED)
    enable_language(CUDA)
endif()

# Include directories
include_directories(include)

# Core library
file(GLOB_RECURSE CORE_SOURCES 
    "src/core/*.cpp"
    "src/frontend/*.cpp" 
    "src/optimization/*.cpp"
    "src/codegen/*.cpp"
)

add_library(neuralcompiler ${CORE_SOURCES})

target_link_libraries(neuralcompiler 
    ${Protobuf_LIBRARIES}
    Threads::Threads
)

if(ENABLE_CUDA)
    target_link_libraries(neuralcompiler ${CUDA_LIBRARIES})
    target_compile_definitions(neuralcompiler PRIVATE ENABLE_CUDA)
endif()

# Examples
add_subdirectory(examples)

# Testing
if(ENABLE_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Benchmarks
if(ENABLE_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()

# Compiler flags
target_compile_options(neuralcompiler PRIVATE
    $<$<CONFIG:Debug>:-g -O0>
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
)

if(ENABLE_SANITIZERS)
    target_compile_options(neuralcompiler PRIVATE -fsanitize=address -fsanitize=undefined)
    target_link_options(neuralcompiler PRIVATE -fsanitize=address -fsanitize=undefined)
endif()

# Install targets
install(TARGETS neuralcompiler
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
)

install(DIRECTORY include/ DESTINATION include)
