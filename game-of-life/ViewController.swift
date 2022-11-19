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

class CommonViewController: UIViewController {
    
    var inProgressFPS: Int {
        60
    }
    var pausedFPS: Int {
        60
    }
    
    var arenaDimensions: CGSize {
        .init(width: 128, height: 128)
    }
    
    var shouldPreserveSquareCells: Bool {
        return true
    }
    
    var colorIndex: Int = 0
    var colors: [CGColor] {
        [.white, .black]
    }
    
    var texturePixelFormat: MTLPixelFormat {
        .rgba8Unorm
    }
    
    private lazy var arenaSize: (width: Int, height: Int) = {
        var output = (width: Int(arenaDimensions.width), height: Int(arenaDimensions.height))
        guard shouldPreserveSquareCells else {
            return output
        }
        let windowSize = UIApplication.shared.keyWindow!.bounds.size
        if windowSize.width > windowSize.height {
            output.width *= 1
            let measure = windowSize.width / CGFloat(output.width)
            output.height = Int(windowSize.height / measure)
        }
        else if windowSize.width < windowSize.height {
            let measure = windowSize.height / CGFloat(output.height)
            output.width = Int(windowSize.width / measure)
            output.height *= 1
        }
        return output
    }()
    
    private(set) lazy var context = try! MTLContext()
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
        view.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch)))
        
        let scrollWheelGesture = UIPanGestureRecognizer(target: self, action: #selector(scrollWheel(gesture:)))
        scrollWheelGesture.allowedScrollTypesMask = .all
        scrollWheelGesture.maximumNumberOfTouches = 0
        view.addGestureRecognizer(scrollWheelGesture)
        view.addGestureRecognizer(UIPinchGestureRecognizer(target: self, action: #selector(pinch)))
        return view
    }()
    private(set) lazy var copy = CopyKernel(context: context)
    private lazy var fill = FillKernel(context: context)
    
    private(set) var cgContext: CGContext!
    private var buffer: MTLBuffer!
    private(set) var texture: MTLTexture!
    
    private var preferredFPS: Int {
        isGamePaused ? pausedFPS : inProgressFPS
    }
    
    private(set) var isGamePaused = true {
        didSet {
            mtkView.preferredFramesPerSecond = preferredFPS
            title = titleText
        }
    }
    private var titleText: String {
        isGamePaused ? "Paused" : "In Progress"
    }
    
    override func viewDidLoad() {
        _ = arenaSize
        super.viewDidLoad()
        
        title = titleText
        navigationItem.rightBarButtonItem = UIBarButtonItem(image: .actions, primaryAction: nil, menu: nil)
        let actions = [
            UIAction(title: "Fill with randoms", handler: { [weak self] _ in
                self?.fillArenaWithRandoms()
            })
        ]
        let menu = UIMenu(title: "Menu", children: actions)
        navigationItem.rightBarButtonItem?.menu = menu
        
        view.addSubview(mtkView)
        
        let width = arenaSize.width
        let height = arenaSize.height
        let pixelRowAlignment = context.device.minimumTextureBufferAlignment(for: .r8Unorm)
        let bytesPerRow = aligned(size: width * 4, align: pixelRowAlignment)

        let pagesize = Int(getpagesize())
        let dataSize = aligned(size: bytesPerRow * height, align: pagesize)
        var data: UnsafeMutableRawPointer?
        posix_memalign(&data, pagesize, dataSize)
        
        self.cgContext = CGContext(
            data: data,
            width: width,
            height: height,
            bitsPerComponent: 8,
            bytesPerRow: bytesPerRow,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
        )
        cgContext.scaleBy(x: 1.0, y: -1.0)
        cgContext.translateBy(x: 0, y: -CGFloat(cgContext.height))
        fill(cgContext: cgContext, dataSize: dataSize)
        
        self.buffer = context.device.makeBuffer(
            bytesNoCopy: cgContext.data!,
            length: dataSize,
            options: .storageModeShared,
            deallocator: { _, _ in
                free(data)
            }
        )
        
        let textureDescriptor = MTLTextureDescriptor()
        textureDescriptor.pixelFormat = texturePixelFormat
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
    
    func fill(cgContext: CGContext, dataSize: Int) {
        cgContext.setFillColor(colors.last!)
        cgContext.fill(CGRect(origin: .zero, size: .init(width: cgContext.width, height: cgContext.height)))
    }
    
    override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        let bounds = view.safeAreaBounds
        let minMeasure = min(bounds.width, bounds.height)
        let size = CGSize(width: minMeasure, height: minMeasure)
        mtkView.frame.origin = bounds.origin
        mtkView.frame.size = bounds.size
    }
    
    func handleTap(tapLocation: CGPoint) {
        let color: CGColor = {
            let x = Int(tapLocation.x)
            let y = Int(tapLocation.y)
            var tapColor: UInt32 = 0
            texture.getBytes(&tapColor, bytesPerRow: texture.bufferBytesPerRow, from: MTLRegion(origin: MTLOrigin(x: x, y: y, z: 0), size: .init(width: 1, height: 1, depth: 1)), mipmapLevel: 0)
#warning("idk why, but we should reverse bytes")
#warning("we can write to texture directly here")
            tapColor = tapColor.cgColor.uint32r
            if tapColor == colors[colorIndex].uint32 {
                colorIndex += 1
                colorIndex %= colors.count
                return colors[colorIndex]
            }
            else {
                print(colorIndex)
                return colors[colorIndex]
            }
        }()
        
        cgContext.setFillColor(color)
        cgContext.fill(CGRect(origin: tapLocation, size: .init(width: 1, height: 1)))
    }
    
    func handlePan(location: CGPoint) {
        cgContext.setFillColor(colors[colorIndex])
        cgContext.fill(CGRect(origin: location, size: .init(width: 1, height: 1)))
    }
    
    @objc private func tap(gesture: UITapGestureRecognizer) {
        let tapLocation = arenaLocation(from: gesture.location(in: mtkView))
        handleTap(tapLocation: tapLocation)
    }
    
    @objc private func longPress(gesture: UILongPressGestureRecognizer) {
        guard gesture.state == .began else {
            return
        }
        isGamePaused.toggle()
    }
    
    @objc private func pan(gesture: UIPanGestureRecognizer) {
        let tapLocation = arenaLocation(from: gesture.location(in: mtkView))
        handlePan(location: tapLocation)
    }
    
    @objc private func pinch(gesture: UIPinchGestureRecognizer) {
        print(#function, gesture.scale)
    }
    
    private(set) var zoomScale: Float = 1.0
    private(set) var zoomTarget = vector_float2(x: 0.5, y: 0.5)
    
    private func updateZoomScale(zoomIn: Bool) {
        zoomScale += 0.01 * (zoomIn ? 1 : -1)
        zoomScale = max(zoomScale, 1)
    }
    
    private func updateZoomTarget(location: CGPoint) {
        let size = vector_float2(x: .init(view.bounds.width), y: .init(view.bounds.height))
        let location = vector_float2(x: .init(location.x), y: .init(location.y)) / size
        let direction = normalize(location - zoomTarget)
        print(direction)
        zoomTarget += direction * 0.001
    }
    
    @objc private func scrollWheel(gesture: UIPanGestureRecognizer) {
        guard gesture.state != .ended else {
            return
        }
//        let delta = gesture.translation(in: view)
//        let location = gesture.location(in: view)
//        updateZoomScale(zoomIn: delta.y > 0)
//        updateZoomTarget(location: location)
    }
    
    private func arenaLocation(from location: CGPoint) -> CGPoint {
        var location = location
        location.x /= mtkView.bounds.width
        location.y /= mtkView.bounds.height
        location.x *= .init(arenaSize.width)
        location.y *= .init(arenaSize.height)
        location.x = floor(location.x)
        location.y = floor(location.y)
        return location
    }
    
    private func fillArenaWithRandoms() {
        for x in 0..<arenaSize.width {
            for y in 0..<arenaSize.height {
                cgContext.setFillColor(colors.randomElement()!)
                cgContext.fill(CGRect(origin: .init(x: x, y: y), size: .init(width: 1, height: 1)))
            }
        }
    }
    
    func encode(commandBuffer: MTLCommandBuffer, destinationTexture: MTLTexture) {
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
            encode(commandBuffer: commandBuffer, previousState: tmpTexture, newState: texture)
        }

        copy.zoomScale = 1.0 / .init(zoomScale)
        copy.zoomTarget = .init(x: .init(zoomTarget.x), y: .init(zoomTarget.y))
        copy.encode(commandBuffer: commandBuffer, sourceTexture: texture, destinationTexture: destinationTexture)

    }
    
    func encode(
        commandBuffer: MTLCommandBuffer,
        previousState: MTLTexture,
        newState: MTLTexture
    ) {
        
    }
}

extension CommonViewController: MTKViewDelegate {
    func mtkView(_ view: MTKView, drawableSizeWillChange size: CGSize) {
        print(#function, size)
    }
    
    func draw(in view: MTKView) {
        guard let commandBuffer = context.commandQueue.makeCommandBuffer(),
              let drawable = view.currentDrawable else {
            print(#function, "NO BUFF AND DRAWABLE")
            return
        }
        let destinationTexture = drawable.texture
        encode(commandBuffer: commandBuffer, destinationTexture: destinationTexture)
        
        commandBuffer.present(drawable)
        commandBuffer.commit()

    }
}

extension UIView {
    var safeAreaBounds: CGRect {
        let origin = CGPoint(x: safeAreaInsets.left, y: safeAreaInsets.top)
        let size = CGSize(width: bounds.width - safeAreaInsets.horizontalInsets,
                          height: bounds.height - safeAreaInsets.verticalInsets)
        return .init(origin: origin, size: size)

    }
}

extension UIEdgeInsets {
    var verticalInsets: CGFloat {
        top + bottom
    }
    
    var horizontalInsets: CGFloat {
        left + right
    }
}

extension CGContext {
    func fillPixel(at location: CGPoint) {
        fill(CGRect(origin: location, size: .init(width: 1, height: 1)))
    }
}

extension CGColor {
    var uint32: UInt32 {
        guard let components,
              components.count == 4 else {
            fatalError()
        }
        let red =   UInt32(components[0].rounded()) * 0xff
        let green = UInt32(components[1].rounded()) * 0xff
        let blue =  UInt32(components[2].rounded()) * 0xff
        let alpha = UInt32(components[3].rounded()) * 0xff
        
        return (red << 24) | (green << 16) | (blue << 8) | (alpha << 0)
    }
    
    var uint32r: UInt32 {
        guard let components,
              components.count == 4 else {
            fatalError()
        }
        let red =   UInt32(components[3].rounded()) * 0xff
        let green = UInt32(components[2].rounded()) * 0xff
        let blue =  UInt32(components[1].rounded()) * 0xff
        let alpha = UInt32(components[0].rounded()) * 0xff
        
        return (red << 24) | (green << 16) | (blue << 8) | (alpha << 0)
    }
    
    static var white: CGColor {
        .init(red: 1, green: 1, blue: 1, alpha: 1)
    }
    static var black: CGColor {
        .init(red: 0, green: 0, blue: 0, alpha: 1)
    }
    static var red: CGColor {
        .init(red: 1, green: 0, blue: 0, alpha: 1)
    }
    static var yellow: CGColor {
        .init(red: 1, green: 1, blue: 0, alpha: 1)
    }
    static var blue: CGColor {
        .init(red: 0, green: 0, blue: 1, alpha: 1)
    }
    static var green: CGColor {
        .init(red: 0, green: 1, blue: 1, alpha: 1)
    }
}

extension UInt32 {
    var cgColor: CGColor {
        let r = CGFloat((self >> 24) & 0xff) / 0xff
        let g = CGFloat((self >> 16) & 0xff) / 0xff
        let b = CGFloat((self >> 8) & 0xff) / 0xff
        let a = CGFloat((self >> 0) & 0xff) / 0xff
        return .init(red: r, green: g, blue: b, alpha: a)
    }
    
    var alphaRemoved: Self {
        self & 0xffffff00
    }
}

extension MTLTexture {
    func getRGBA(x: Int, y: Int) -> UInt32 {
        guard x < width, y < height else {
            return 0
        }
        var rgba: UInt32 = 0
        getBytes(
            &rgba,
            bytesPerRow: bufferBytesPerRow,
            from: pixelRegion(x: x, y: y),
            mipmapLevel: 0)
        return rgba
    }
    
    func write(rgba: UInt32, x: Int, y: Int) {
        guard x < width, y < height else {
            return
        }
        var rgba = rgba
        replace(
            region: pixelRegion(x: x, y: y),
            mipmapLevel: 0,
            withBytes: &rgba,
            bytesPerRow: bufferBytesPerRow
        )
    }
    
    private func pixelRegion(x: Int, y: Int) -> MTLRegion {
        MTLRegion(origin: .init(x: x, y: y, z: 0), size: .init(width: 1, height: 1, depth: 1))
    }
}
