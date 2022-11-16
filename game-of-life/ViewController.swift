//
//  ViewController.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/15/22.
//

import UIKit
import MetalKit
import MetalPerformanceShaders

func aligned(size: Int, align: Int) -> Int {
    let count = size / align + (((size % align) != 0) ? 1 : 0)
    return count * align
}

class ViewController: UIViewController {
    
    let arenaMeasure = 64
    
    private lazy var context = try! MTLContext()
    private lazy var mtkView: MTKView = {
        let view = MTKView()
        view.clearColor = .init(red: 0, green: 1.0, blue: 0, alpha: 1.0)
        view.device = context.device
        view.delegate = self
        view.framebufferOnly = false
        view.preferredFramesPerSecond = preferredFPS
        
        view.addGestureRecognizer(UITapGestureRecognizer(target: self, action: #selector(tap)))
        view.addGestureRecognizer(UILongPressGestureRecognizer(target: self, action: #selector(longPress)))
        view.addGestureRecognizer(UIPanGestureRecognizer(target: self, action: #selector(pan)))
        return view
    }()
    private lazy var copy = CopyKernel(context: context)
    private lazy var fill = FillKernel(context: context)
    private lazy var gol = GOLKernel(context: context)
    private lazy var arenaTexture: MTLTexture = {
        let descriptor = MTLTextureDescriptor.texture2DDescriptor(
            pixelFormat: .bgra8Unorm,
            width: arenaMeasure,
            height: arenaMeasure,
            mipmapped: false
        )
        descriptor.usage = [.shaderRead, .shaderWrite]
        return context.device.makeTexture(descriptor: descriptor)!
    }()
    
    private var cgContext: CGContext!
    private var buffer: MTLBuffer!
    private var texture: MTLTexture!
    
    private var preferredFPS: Int {
        isGamePaused ? 60 : 10
    }
    
    private var isGamePaused = true {
        didSet {
            mtkView.preferredFramesPerSecond = preferredFPS
        }
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        view.addSubview(mtkView)
        
        let width = arenaMeasure
        let height = arenaMeasure
        let pixelRowAlignment = context.device.minimumTextureBufferAlignment(for: .r8Unorm)
        let bytesPerRow = aligned(size: width, align: pixelRowAlignment)

        let pagesize = Int(getpagesize())
        let dataSize = aligned(size: bytesPerRow * height, align: pagesize)
        var data: UnsafeMutableRawPointer?
        posix_memalign(&data, pagesize, dataSize)
        memset(data, 0, dataSize)
        
        self.cgContext = CGContext(
            data: data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGImageAlphaInfo.none.rawValue
        )
        cgContext.scaleBy(x: 1.0, y: -1.0)
        cgContext.translateBy(x: 0, y: -CGFloat(cgContext.height))
        
        cgContext.setFillColor(red: 1.0, green: 1.0, blue: 1.0, alpha: 1.0)
        
        self.buffer = context.device.makeBuffer(
            bytesNoCopy: cgContext.data!,
            length: dataSize,
            options: .storageModeShared,
            deallocator: { _, _ in
                free(data)
            }
        )
        
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.pixelFormat = .r8Unorm
        textureDescriptor.width = cgContext.width
        textureDescriptor.height = cgContext.height
        textureDescriptor.storageMode = buffer.storageMode
        textureDescriptor.usage = [.shaderRead, .shaderWrite]
        
        self.texture = buffer.makeTexture(
            descriptor: textureDescriptor,
            offset: 0,
            bytesPerRow: cgContext.bytesPerRow
        )
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        let minMeasure = min(view.bounds.width, view.bounds.height)
        let size = CGSize(width: minMeasure, height: minMeasure)
        mtkView.frame = CGRect(origin: .zero, size: size)
    }
    
    @objc private func tap(gesture: UITapGestureRecognizer) {
        let tapLocation = arenaLocation(from: gesture.location(in: mtkView))
        
        let color: CGColor = {
            let x = Int(tapLocation.x)
            let y = Int(tapLocation.y)
            var value: UInt8 = 0
            texture.getBytes(&value, bytesPerRow: texture.bufferBytesPerRow, from: MTLRegion(origin: MTLOrigin(x: x, y: y, z: 0), size: .init(width: 1, height: 1, depth: 1)), mipmapLevel: 0)
            let isLive = value > 0
            return CGColor(gray: isLive ? 0 : 1, alpha: 1)
        }()
        
        cgContext.setFillColor(color)
        cgContext.fill(CGRect(origin: tapLocation, size: .init(width: 1, height: 1)))
    }
    
    @objc private func longPress(gesture: UILongPressGestureRecognizer) {
        guard gesture.state == .began else {
            return
        }
        isGamePaused.toggle()
    }
    
    @objc private func pan(gesture: UIPanGestureRecognizer) {
        let tapLocation = arenaLocation(from: gesture.location(in: mtkView))
        cgContext.setFillColor(CGColor(gray: 1, alpha: 1))
        cgContext.fill(CGRect(origin: tapLocation, size: .init(width: 1, height: 1)))
    }
    
    private func arenaLocation(from location: CGPoint) -> CGPoint {
        var location = location
        location.x /= mtkView.bounds.width
        location.y /= mtkView.bounds.height
        location.x *= .init(arenaMeasure)
        location.y *= .init(arenaMeasure)
        location.x = floor(location.x)
        location.y = floor(location.y)
        return location
    }
}

extension ViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        print(#function, size)
    }
    
    func draw(in view: MTKView) {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let drawable = view.currentDrawable else {
            print(#function, "NO BUFF AND DRAWABLE")
            return
        }
        
        if isGamePaused == false {
            let tmpImage = MPSTemporaryImage(
                commandBuffer: commandBuffer,
                textureDescriptor: texture.temporaryImageDescriptor
            )
            defer {
                tmpImage.readCount = 0
            }
            let tmpTexture = tmpImage.texture
            let blit = commandBuffer.makeBlitCommandEncoder()!
            blit.copy(from: texture, to: tmpTexture)
            blit.endEncoding()
            gol.encode(commandBuffer: commandBuffer, sourceTexture: tmpTexture, destinationTexture: texture)
        }

        let drawableTexture = drawable.texture
//        fill.encode(commandBuffer: commandBuffer, destinationTexture: arenaTexture)
        copy.encode(commandBuffer: commandBuffer, sourceTexture: texture, destinationTexture: drawableTexture)

        commandBuffer.present(drawable)
        commandBuffer.commit()

    }
}
