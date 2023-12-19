import Foundation
import UIKit
import MetalPerformanceShaders

final class SmoothLifeViewController: CommonViewController {
    override var inProgressFPS: Int {
        60
    }

    override var arenaDimensions: CGSize {
        .init(width: 1024, height: 1024)
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        isHoverEnabled = false
        isZoomEnabled = false
        fillArenaWithRandoms()
//        isGamePaused = false
    }

    private lazy var smoothLifeComputePipelineState: MTLComputePipelineState = {
        try! context.makeComputePipelineState(functionName: "smoothlife")
    }()

    private lazy var smoothLife2ComputePipelineState: MTLComputePipelineState = {
        try! context.makeComputePipelineState(functionName: "smoothlife2")
    }()

    override func encode(
        commandBuffer: MTLCommandBuffer,
        previousState: MTLTexture,
        newState: MTLTexture
    ) {
        let use2 = false

        assert(previousState.size.width == newState.size.width)
        assert(previousState.size.height == newState.size.height)
        assert(previousState.size.depth == newState.size.depth)

        if use2 {
            let bigBlurImage = MPSTemporaryImage(commandBuffer: commandBuffer, textureDescriptor: previousState.temporaryImageDescriptor)
            let smallBlurImage = MPSTemporaryImage(commandBuffer: commandBuffer, textureDescriptor: previousState.temporaryImageDescriptor)
            defer {
                bigBlurImage.readCount = 0
                smallBlurImage.readCount = 0
            }
            let bigBlurSigma: Float = 21.0
            let bigBlur = MPSImageGaussianBlur(device: context.device, sigma: bigBlurSigma)
            let smallBlur = MPSImageGaussianBlur(device: context.device, sigma: bigBlurSigma / 3)

            bigBlur.encode(commandBuffer: commandBuffer, sourceTexture: previousState, destinationTexture: bigBlurImage.texture)
            smallBlur.encode(commandBuffer: commandBuffer, sourceTexture: previousState, destinationTexture: smallBlurImage.texture)

            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.set(textures: [previousState, bigBlurImage.texture, smallBlurImage.texture, newState])
            encoder.dispatch2d(state: smoothLife2ComputePipelineState, size: newState.size)
            encoder.endEncoding()
        } else {
            print(previousState.size, newState.size)
            let encoder = commandBuffer.makeComputeCommandEncoder()!
            encoder.set(textures: [previousState, newState])
            encoder.dispatch2d(state: smoothLifeComputePipelineState, size: newState.size)
            encoder.endEncoding()
        }
    }
}

