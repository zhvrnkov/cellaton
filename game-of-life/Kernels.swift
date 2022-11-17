//
//  Kernels.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/15/22.
//

import Foundation
import MetalPerformanceShaders
import CoreMedia

class UnaryImageKernel: MPSUnaryImageKernel {
    class var kernelName: String {
        ""
    }
    let context: MTLContext
    var param: Float = 1.0
    var time: CMTime = .zero
    private(set) var prevTexture: MTLTexture?
    private(set) lazy var pipelineState = try! context.makeComputePipelineState(functionName: Self.kernelName)
    
    init(context: MTLContext) {
        self.context = context
        super.init(device: context.device)
    }
    
    required init?(coder aDecoder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    override func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.set(textures: [sourceTexture, destinationTexture])
        encoder.dispatch2d(state: pipelineState, size: destinationTexture.size)
        encoder.endEncoding()
    }
    
    func prevTexture(d: MTLTextureDescriptor) -> MTLTexture {
        if prevTexture?.descriptor.width != d.width || prevTexture?.descriptor.height != d.height {
            print("nil")
            prevTexture = context.device.makeTexture(descriptor: d)
        }
        return self.prevTexture!
    }
    
    func savePrev(texture: MTLTexture, commandBuffer: MTLCommandBuffer) {
        let blit = commandBuffer.makeBlitCommandEncoder()!
        blit.copy(from: texture, to: prevTexture(d: texture.descriptor))
        blit.endEncoding()
    }
}

final class CopyKernel: UnaryImageKernel {
    override class var kernelName: String {
        "copy"
    }
    
    override func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.set(textures: [sourceTexture, destinationTexture])
        encoder.dispatch2d(state: pipelineState, size: destinationTexture.size)
        encoder.endEncoding()
    }
}

final class FillKernel: UnaryImageKernel {
    override class var kernelName: String {
        "fill"
    }
    func encode(commandBuffer: MTLCommandBuffer, destinationTexture: MTLTexture) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        var color = simd_float4(x: 1.0, y: 0, z: 0, w: 1.0)
        encoder.set(value: &color, index: 0)
        encoder.set(textures: [destinationTexture])
        let size = destinationTexture.size
        encoder.dispatch2d(state: pipelineState, size: .init(width: size.width, height: 1, depth: size.depth))
        encoder.endEncoding()
    }
}

extension Bool: ExpressibleByIntegerLiteral {
    public typealias IntegerLiteralType = Int

    public init(integerLiteral value: Int) {
        self = !(value == 0)
    }
}

typealias Grid = [[Int32]]
final class GOLKernel: UnaryImageKernel {
    override class var kernelName: String {
        "gol"
    }
    
    let grid: Grid = .moore1
    private lazy var flatGrid = grid.reduce(Grid.Element()) { $0 + $1 }
    lazy var liveActivations: [vector_float4] = .seedsLiveActivations(gridLength: grid.length)
    lazy var deadActivations: [vector_float4] = .seedsDeadActivations(gridLength: grid.length)
    
    override func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.set(textures: [sourceTexture, destinationTexture])
        let gridHeight = grid.count
        let gridWidth = grid[0].count
        var gridDim = vector_int2(x: Int32(gridWidth), y: Int32(gridHeight))
        var flatGrid = grid.reduce([Int32]()) { $0 + $1 }
        let gridLength = MemoryLayout.stride(ofValue: flatGrid[0]) * flatGrid.count
        encoder.setBytes(&flatGrid, length: gridLength, index: 0)
        encoder.set(value: &gridDim, index: 1)
        encoder.set(array: &liveActivations, index: 2)
        encoder.set(array: &deadActivations, index: 3)
        encoder.dispatch2d(state: pipelineState, size: destinationTexture.size)
        encoder.endEncoding()
    }
}

extension Grid {
    
    var length: Int {
        count * self[0].count
    }
    
    static var moore1: Grid {
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ]
    }
    
    static var moore2: Grid {
        [
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ]
    }
    
    static var vonNeumann1: Grid {
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
        ]
    }
    
    static var vonNeumann2: Grid {
        [
            [0, 0, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 0, 0],
        ]
    }
}

typealias Rule = [vector_float4]
final class RuleKernel: UnaryImageKernel {
    
    override class var kernelName: String {
        "row_fill"
    }
    
    var rule: Rule = []
    
    func encode(commandBuffer: MTLCommandBuffer, destinationTexture: MTLTexture) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        var offset = simd_int3(x: Int32(offset.x), y: Int32(offset.y), z: Int32(offset.z))
        var bitsCount = Int32(log2(Float(rule.count)))
        encoder.set(value: &offset, index: 0)
        encoder.setBytes(&rule, length: MemoryLayout.stride(ofValue: rule[0]) * rule.count, index: 1)
        encoder.set(value: &bitsCount, index: 2)
        encoder.set(textures: [destinationTexture])
        let size = destinationTexture.size
        encoder.dispatch2d(state: pipelineState, size: .init(width: size.width, height: 1, depth: size.depth))
        encoder.endEncoding()
    }
}

typealias f4 = vector_float4

extension f4 {
    init(_ r: Float, _ g: Float, _ b: Float, _ a: Float = 1.0) {
        self.init(x: r, y: g, z: b, w: a)
    }
}

extension Rule {
    static var rule90: Rule {
        rule([
            (0b110, f4(1, 0, 1)),
            (0b100, f4(0, 1, 0)),
            (0b011, f4(0, 0, 1)),
            (0b001, f4(1, 0, 0))
        ], 3)
    }
    
    static var rule110: Rule {
        rule([
            (0b110, f4(1, 0, 0)),
            (0b101, f4(0, 1, 0)),
            (0b011, f4(0, 0, 1)),
            (0b010, f4(1, 1, 0)),
            (0b001, f4(0, 1, 1))
        ], 3)
    }
    
    static var rule30: Rule {
        rule([
            (0b100, f4(1, 0, 0)),
            (0b011, f4(0, 1, 0)),
            (0b010, f4(0, 0, 1)),
            (0b001, f4(1, 0, 1))
        ], 3)
    }
    
    static func convayLiveActivations(gridLength: Int) -> Rule {
        rule([
            (2, f4(1,0,0)),
            (3, f4(0,1,0))
        ], gridLength)
    }

    static func convayDeadActivations(gridLength: Int) -> Rule {
        rule([
            (3, f4(0,0,1))
        ], gridLength)
    }
    
    static func seedsLiveActivations(gridLength: Int) -> Rule {
        rule([], gridLength)
    }
    
    static func seedsDeadActivations(gridLength: Int) -> Rule {
        rule([(2, f4(1,1,1))], gridLength)
    }
}

extension Array {
    static func rule(
        _ activations: [(index: Int, color: vector_float4)],
        nob: Int,
        defaultColor: vector_float4 = vector_float4(x: 0, y: 0, z: 0, w: 1)
    ) -> [vector_float4] {
        return rule(activations, 1 << nob)
    }

    static func rule(
        _ activations: [(index: Int, color: vector_float4)],
        _ count: Int,
        defaultColor: vector_float4 = vector_float4(x: 0, y: 0, z: 0, w: 1)
    ) -> [vector_float4] {
        var output = [vector_float4](repeating: defaultColor, count: count)
        for (index, color) in activations {
            output[index] = color
        }
        return output
    }
}
