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

final class GOLKernel: UnaryImageKernel {
    override class var kernelName: String {
        "gol"
    }
    
    override func encode(commandBuffer: MTLCommandBuffer, sourceTexture: MTLTexture, destinationTexture: MTLTexture) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.set(textures: [sourceTexture, destinationTexture])
        encoder.dispatch2d(state: pipelineState, size: destinationTexture.size)
        encoder.endEncoding()
    }
}

final class RowKernel: UnaryImageKernel {
    override class var kernelName: String {
        "row_fill"
    }
    
    var activations: [vector_float4] = .rule90
    
    func encode(commandBuffer: MTLCommandBuffer, destinationTexture: MTLTexture) {
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        var offset = simd_int3(x: Int32(offset.x), y: Int32(offset.y), z: Int32(offset.z))
        var bitsCount = Int32(log2(Float(activations.count)))
        encoder.set(value: &offset, index: 0)
        encoder.setBytes(&activations, length: MemoryLayout.stride(ofValue: activations[0]) * activations.count, index: 1)
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

extension Array {
    static var rule90: [vector_float4] {
        rule([
            (0b110, f4(1, 0, 1)),
            (0b100, f4(0, 1, 0)),
            (0b011, f4(0, 0, 1)),
            (0b001, f4(1, 0, 0))
        ], 3)
    }
    
    static var rule110: [vector_float4] {
        rule([
            (0b110, f4(1, 0, 0)),
            (0b101, f4(0, 1, 0)),
            (0b011, f4(0, 0, 1)),
            (0b010, f4(1, 1, 0)),
            (0b001, f4(0, 1, 1))
        ], 3)
    }
    
    static var rule30: [vector_float4] {
        rule([
            (0b100, f4(1, 0, 0)),
            (0b011, f4(0, 1, 0)),
            (0b010, f4(0, 0, 1)),
            (0b001, f4(1, 0, 1))
        ], 3)
    }
    
    static func rule(
        _ activations: [(index: Int, color: vector_float4)],
        _ numberOfBits: Int,
        defaultColor: vector_float4 = vector_float4(x: 0, y: 0, z: 0, w: 1)
    ) -> [vector_float4] {
        var output = [vector_float4](repeating: defaultColor, count: 1 << numberOfBits)
        for (index, color) in activations {
            output[index] = color
        }
        return output
    }
}
