//
//  HPPLatticeGasViewController.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/19/22.
//

import Foundation
import Metal
import MetalPerformanceShaders
import AVFoundation

fileprivate let wall: UInt32  = 0b100000000
//fileprivate let top: UInt32   = 0b001 << (3 * 3)
//fileprivate let right: UInt32 = 0b011 << (3 * 2)
//fileprivate let bot: UInt32   = 0b101 << (3 * 1)
//fileprivate let left: UInt32  = 0b111 << (3 * 0)

final class HPPLaticeGasViewController: CommonViewController {
    
    override var inProgressFPS: Int {
       60
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 4096, height: 4096)
    }
    
    override var shouldPreserveSquareCells: Bool {
        true
    }
    
    override var texturePixelFormat: MTLPixelFormat {
        .r32Uint
    }
    
    override var colors: [CGColor] {
        [.init(red: 0, green: 0, blue: 0, alpha: 0.0)]
    }
    
    private lazy var latticeGas = LatticeGasKernel(context: context)
    private lazy var latticeGasFillRect = LatticeGasFillRectKernel(context: context)
    private lazy var latticeGasFillCircle = LatticeGasFillCircleKernel(context: context)
    private lazy var latticeGas2Image = LaticeGasToImageKernel(context: context)
    
    private lazy var tmpTexture: MTLTexture = {
        context.device.makeTexture(descriptor: texture.descriptor)!
    }()
    
    override func fill(cgContext: CGContext, dataSize: Int) {
        memset(cgContext.data, 0, dataSize)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()

        let width = cgContext.width
        let height = cgContext.height
        let centerX = width / 2 - 1
        let centerY = height / 2 - 1
        
        let quarterH = height / 4
        let halfWidth = width / 2
        let low = halfWidth - quarterH
        let high = halfWidth + quarterH
        
        let commandBuffer = context.commandQueue.makeCommandBuffer()!
        latticeGasFillRect.percent = 0.5
        latticeGasFillRect.encode(
            commandBuffer: commandBuffer,
            destinationTexture: texture,
            rect: CGRect(origin: .init(x: 0, y: 0), size: .init(width: 1, height: 1))
        )
        latticeGasFillCircle.percent = 1.0
        latticeGasFillCircle.encode(
            commandBuffer: commandBuffer,
            destinationTexture: texture,
            radius: 0.25,
            center: .init(x: 0.5, y: 0.5)
        )

        latticeGasFillRect.percent = 1.0
        latticeGasFillRect.encode(
            commandBuffer: commandBuffer,
            destinationTexture: texture,
            rect: CGRect(origin: .init(x: 0.5 - 0.1, y: 0.5 - 0.1), size: .init(width: 0.2, height: 0.2))
        )
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
       
        let windowWidth = Int(Float(width) * 0.3)
        let windowHeight = Int(Float(height) * 0.3)
        let windowX = width / 10
        let windowY = height / 10
        
        let regions: [MTLRegion] = [
            MTLRegion(origin: .init(x: 0, y: 0, z: 0), size: .init(width: width, height: 2, depth: 1)), // top
            MTLRegion(origin: .init(x: 0, y: height - 2, z: 0), size: .init(width: width, height: 2, depth: 1)), // bot
            MTLRegion(origin: .init(x: 0, y: 0, z: 0), size: .init(width: 2, height: height, depth: 1)), // left
            MTLRegion(origin: .init(x: width - 2, y: 0, z: 0), size: .init(width: 2, height: height - 1, depth: 1)), // right
        ]
        
        for region in regions {
            let count = region.size.area
            var bytes: [UInt32] = Array(repeating: wall, count: count)
            texture.replace(region: region, mipmapLevel: 0, withBytes: &bytes, bytesPerRow: region.size.width * 4)
        }
        
        texture.write(rgba: 0b1000, x: centerX, y: centerY + 2)
        texture.write(rgba: 0b0010, x: centerX, y: centerY - 2)
        texture.write(rgba: 0b0100, x: centerX - 2, y: centerY)
        texture.write(rgba: 0b0001, x: centerX + 2, y: centerY)

        
//        spawnWindow(x: windowX, y: windowY, w: windowWidth, h: windowHeight)
//        spawnWindow(x: width - windowX - windowWidth, y: windowY, w: windowWidth, h: windowHeight)
//        spawnWindow(x: width - windowX - windowWidth, y: height - windowHeight - windowY, w: windowWidth, h: windowHeight)
//        spawnWindow(x: windowX, y: height - windowHeight - windowY, w: windowWidth, h: windowHeight)
//
//        let shieldRegion = MTLRegion(origin: .init(x: width / 2, y: height / 3, z: 0),
//                                     size: .init(width: 4, height: height / 3, depth: 1))
//        var bytes: [UInt32] = Array(repeating: wall, count: shieldRegion.size.area)
//        texture.replace(region: shieldRegion, mipmapLevel: 0, withBytes: &bytes, bytesPerRow: shieldRegion.size.width * 4)
    }
    
    private func spawnRandomWindow() {
        let width = Int(Float.random(in: 0.01...0.1) * Float(texture.width))
        let height = Int(Float.random(in: 0.01...0.1) * Float(texture.height))
        var x = Int(Float.random(in: 0...1) * Float(texture.width))
        var y = Int(Float.random(in: 0...1) * Float(texture.height))
        
        if (width + x) > texture.width {
            x -= width
        }
        if (height + y) > texture.height {
            y -= height
        }
        
        spawnWindow(x: x, y: y, w: width, h: height)
    }
    
    private func spawnWindow(x: Int, y: Int, w: Int, h: Int) {
        let windowRegion = MTLRegion(origin: .init(x: x, y: y, z: 0),
                                     size: .init(width: w, height: h, depth: 1))
        let windowBytesCount = windowRegion.size.area
        var windowBytes = Array(repeating: UInt32(0), count: windowBytesCount)
        texture.replace(
            region: windowRegion,
            mipmapLevel: 0,
            withBytes: &windowBytes,
            bytesPerRow: windowRegion.size.width * 4
        )
    }
    
    override func handleTap(tapLocation: CGPoint) {
        texture.write(rgba: wall, x: .init(tapLocation.x), y: .init(tapLocation.y))
        texture.write(rgba: wall, x: .init(tapLocation.x-1), y: .init(tapLocation.y))
        texture.write(rgba: wall, x: .init(tapLocation.x-1), y: .init(tapLocation.y-1))
        texture.write(rgba: wall, x: .init(tapLocation.x), y: .init(tapLocation.y-1))
    }
    
    override func handlePan(location: CGPoint) {
        handleTap(tapLocation: location)
    }
    
    override func encode(
        commandBuffer: MTLCommandBuffer,
        destinationTexture: MTLTexture
    ) {
        if isGamePaused == false {
            let blit = commandBuffer.makeBlitCommandEncoder()!
            blit.copy(from: texture, to: tmpTexture)
            blit.endEncoding()
            
            latticeGas.encode(commandBuffer: commandBuffer, sourceTexture: tmpTexture, destinationTexture: texture)
            latticeGas.encode(commandBuffer: commandBuffer, sourceTexture: texture, destinationTexture: tmpTexture)
            latticeGas.encode(commandBuffer: commandBuffer, sourceTexture: tmpTexture, destinationTexture: texture)
        }
        
        let descriptor = texture.temporaryImageDescriptor
        descriptor.pixelFormat = .rgba8Unorm
        let lg2img = MPSTemporaryImage(commandBuffer: commandBuffer, textureDescriptor: descriptor)
        defer {
            lg2img.readCount = 0
        }
        
        latticeGas2Image.encode(commandBuffer: commandBuffer, sourceTexture: texture, destinationTexture: lg2img.texture)
        
        updateCopyKernel()
        copy.encode(commandBuffer: commandBuffer, sourceTexture: lg2img.texture, destinationTexture: destinationTexture)
    }
    
}

final class LatticeGasKernel: UnaryImageKernel {
    override class var kernelName: String {
        "hppModel"
    }
}

final class LaticeGasToImageKernel: UnaryImageKernel {
    override class var kernelName: String {
        "latticeGasToImage"
    }
}

final class LatticeGasFillCircleKernel: UnaryImageKernel {
    override class var kernelName: String {
        "latticeGasFillCircle"
    }
    
    var percent: Float = 0.5
    var seed: UInt32 = .random(in: UInt32.min...UInt32.max)

    func encode(
        commandBuffer: MTLCommandBuffer,
        destinationTexture: MTLTexture,
        radius: CGFloat,
        center: CGPoint
    ) {
        let dts = destinationTexture.size
        let size = MTLSize(
            width: Int(CGFloat(dts.height) * radius * 2),
            height: Int(CGFloat(dts.height) * radius * 2),
            depth: dts.depth
        )
        var center = simd_uint2(x: .init(CGFloat(dts.width) * center.x), y: .init(CGFloat(dts.height) * center.y))
        var radius = UInt32(CGFloat(dts.height) * radius)
        
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.set(textures: [destinationTexture])
        encoder.set(value: &seed, index: 0)
        encoder.set(value: &percent, index: 1)
        encoder.set(value: &center, index: 2)
        encoder.set(value: &radius, index: 3)
        encoder.dispatch2d(state: pipelineState, size: destinationTexture.size)
        encoder.endEncoding()
    }
}

final class LatticeGasFillRectKernel: UnaryImageKernel {
    override class var kernelName: String {
        "latticeGasFillRect"
    }
    
    var percent: Float = 0.5
    var seed: UInt32 = .random(in: UInt32.min...UInt32.max)
    
    func encode(
        commandBuffer: MTLCommandBuffer,
        destinationTexture: MTLTexture,
        rect: CGRect
    ) {
        let dts = destinationTexture.size
        let size = MTLSize(
            width: Int(CGFloat(dts.width) * rect.width),
            height: Int(CGFloat(dts.height) * rect.height),
            depth: dts.depth
        )
        offset = MPSOffset(
            x: Int(CGFloat(dts.width) * rect.origin.x),
            y: Int(CGFloat(dts.height) * rect.origin.y),
            z: 0
        )
        var offset = simd_int3(x: Int32(offset.x), y: Int32(offset.y), z: Int32(offset.z))
        let encoder = commandBuffer.makeComputeCommandEncoder()!
        encoder.set(textures: [destinationTexture])
        encoder.set(value: &seed, index: 0)
        encoder.set(value: &percent, index: 1)
        encoder.set(value: &offset, index: 2)
        encoder.dispatch2d(state: pipelineState, size: size)
        encoder.endEncoding()
    }
}

extension MTLSize {
    var area: Int {
        width * height * depth
    }
}
