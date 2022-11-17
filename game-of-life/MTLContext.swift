//
//  File.swift
//  core
//
//  Created by Zhavoronkov Vlad on 5/25/22.
//

import Foundation
import Metal
import MetalKit

final class MTLContext {
    enum Error: Swift.Error {
        case noFunction(name: String)
        case noDeviceOrCommandQueue
    }

    let device: MTLDevice
    let commandQueue: MTLCommandQueue
    let library: MTLLibrary
    private(set) lazy var ciContext: CIContext = {
        let options: [CIContextOption: Any] = [.cacheIntermediates: NSNumber(false),
                                               .outputPremultiplied: NSNumber(true),
                                               CIContextOption.useSoftwareRenderer: NSNumber(false),
                                               .workingColorSpace: NSNull()]
        
        return CIContext(mtlCommandQueue: commandQueue, options: options)
    }()

    init() throws {
        guard let device = MTLCreateSystemDefaultDevice(),
              let commandQueue = device.makeCommandQueue() else {
            throw Error.noDeviceOrCommandQueue
        }
        let library = try device.makeDefaultLibrary(bundle: Bundle.main)
        self.device = device
        self.commandQueue = commandQueue
        self.library = library
    }
    
    func renderPipelineState(pixelFormat: MTLPixelFormat, prefix: String?) throws -> MTLRenderPipelineState {
        let prefix = prefix ?? String(describing: self)
        let descriptor = MTLRenderPipelineDescriptor(
            vertexFunction: "\(prefix)_vertexFunction",
            fragmentFunction: "\(prefix)_fragmentFunction",
            pixelFormat: pixelFormat,
            library: library
        )
        return try device.makeRenderPipelineState(descriptor: descriptor)
    }
    
}

extension MTLContext {
    func makeComputePipelineState(functionName: String) throws -> MTLComputePipelineState {
        guard let function = library.makeFunction(name: functionName) else {
            throw Error.noFunction(name: functionName)
        }
        return try device.makeComputePipelineState(function: function)
    }
}

extension MTLRenderPipelineDescriptor {
    convenience init(vertexFunction: String, fragmentFunction: String, pixelFormat: MTLPixelFormat, library: MTLLibrary) {
        self.init()
        self.colorAttachments[0].pixelFormat = pixelFormat
        self.colorAttachments[0].isBlendingEnabled = true
        self.colorAttachments[0].rgbBlendOperation = .add
        self.colorAttachments[0].alphaBlendOperation = .add
        self.colorAttachments[0].sourceRGBBlendFactor = .sourceAlpha
        self.colorAttachments[0].sourceAlphaBlendFactor = .sourceAlpha
        self.colorAttachments[0].destinationRGBBlendFactor = .oneMinusSourceAlpha
        self.colorAttachments[0].destinationAlphaBlendFactor = .oneMinusSourceAlpha
        self.vertexFunction = library.makeFunction(name: vertexFunction)!
        self.fragmentFunction = library.makeFunction(name: fragmentFunction)!
    }
}

public extension MTLComputePipelineState {
    var executionWidthThreadgroupSize: MTLSize {
        let width = self.threadExecutionWidth
        
        return MTLSize(width: width, height: 1, depth: 1)
    }
    
    var max1dThreadgroupSize: MTLSize {
        let width = self.maxTotalThreadsPerThreadgroup
        
        return MTLSize(width: width, height: 1, depth: 1)
    }
    
    var max2dThreadgroupSize: MTLSize {
        let width = self.threadExecutionWidth
        let height = self.maxTotalThreadsPerThreadgroup / width
    
        return MTLSize(width: width, height: height, depth: 1)
    }
}

public enum Feature {
    case nonUniformThreadgroups
    case readWriteTextures(MTLPixelFormat)
}

public extension MTLDevice {
    func supports(feature: Feature) -> Bool {
        switch feature {
        case .nonUniformThreadgroups:
            #if targetEnvironment(macCatalyst)
            return self.supportsFamily(.common3)
            #elseif os(iOS)
            return self.supportsFeatureSet(.iOS_GPUFamily4_v1)
            #elseif os(macOS)
            return self.supportsFeatureSet(.macOS_GPUFamily1_v3)
            #endif
            
        case let .readWriteTextures(pixelFormat):
            let tierOneSupportedPixelFormats: Set<MTLPixelFormat> = [
                .r32Float, .r32Uint, .r32Sint
            ]
            let tierTwoSupportedPixelFormats: Set<MTLPixelFormat> = tierOneSupportedPixelFormats.union([
                .rgba32Float, .rgba32Uint, .rgba32Sint, .rgba16Float,
                .rgba16Uint, .rgba16Sint, .rgba8Unorm, .rgba8Uint,
                .rgba8Sint, .r16Float, .r16Uint, .r16Sint,
                .r8Unorm, .r8Uint, .r8Sint
            ])
            
            switch self.readWriteTextureSupport {
            case .tier1: return tierOneSupportedPixelFormats.contains(pixelFormat)
            case .tier2: return tierTwoSupportedPixelFormats.contains(pixelFormat)
            case .tierNone: return false
            @unknown default: return false
            }
        }
    }
}

extension MTLComputeCommandEncoder {
    func set<T>(value: inout T, index: Int) {
        setBytes(&value, length: MemoryLayout<T>.stride, index: index)
    }
    
    func set<T>(array: inout [T], index: Int) {
        setBytes(&array, length: MemoryLayout<T>.stride * array.count, index: index)
    }
    
    func dispatch2d(state: MTLComputePipelineState, size: MTLSize) {
        if device.supports(feature: .nonUniformThreadgroups) {
            dispatch2d(state: state, exactly: size)
        }
        else {
            dispatch2d(state: state, covering: size)
        }
    }
    
    func dispatch2d(state: MTLComputePipelineState,
                    covering size: MTLSize,
                    threadgroupSize: MTLSize? = nil) {
        let tgSize = threadgroupSize ?? state.max2dThreadgroupSize
        
        let count = MTLSize(width: (size.width + tgSize.width - 1) / tgSize.width,
                            height: (size.height + tgSize.height - 1) / tgSize.height,
                            depth: 1)
        
        self.setComputePipelineState(state)
        self.dispatchThreadgroups(count, threadsPerThreadgroup: tgSize)
    }
    
    func dispatch2d(state: MTLComputePipelineState,
                    exactly size: MTLSize,
                    threadgroupSize: MTLSize? = nil) {
        let tgSize = threadgroupSize ?? state.max2dThreadgroupSize
        
        self.setComputePipelineState(state)
        self.dispatchThreads(size, threadsPerThreadgroup: tgSize)
    }
    
    func set(textures: [MTLTexture]) {
        setTextures(textures, range: textures.indices)
    }
}

extension MTLTexture {
    var size: MTLSize {
        MTLSize(width: width, height: height, depth: depth)
    }

    var descriptor: MTLTextureDescriptor {
        let output = MTLTextureDescriptor()
        
        output.width = width
        output.height = height
        output.depth = depth
        output.arrayLength = arrayLength
        output.storageMode = storageMode
        output.cpuCacheMode = cpuCacheMode
        output.usage = usage
        output.textureType = textureType
        output.sampleCount = sampleCount
        output.mipmapLevelCount = mipmapLevelCount
        output.pixelFormat = pixelFormat
        output.allowGPUOptimizedContents = allowGPUOptimizedContents

        return output
    }
    
    var temporaryImageDescriptor: MTLTextureDescriptor {
        let descriptor = self.descriptor
        descriptor.storageMode = .private
        return descriptor
    }
}
