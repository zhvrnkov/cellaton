//
//  GameOfLifeViewController.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/17/22.
//

import Foundation
import Metal
import CoreGraphics
import MetalKit

class GameOfLifeViewController: CommonViewController {
    override var inProgressFPS: Int {
        60
    }
    
    var setup: Setup {
        .init(
            gridType: .moore1,
            kind: .wireWorld
        )
    }
    private lazy var gol: GOLKernel = {
        let kernel = GOLKernel(context: context)
        let (grid, rules, table) = setup.data
        kernel.setup(grid: grid, rules: rules, stateToDelta: table)
        return kernel
    }()
    
    override var arenaDimensions: CGSize {
        .init(width: 128, height: 128)
    }
    
    override var shouldPreserveSquareCells: Bool {
        false
    }

    override func encode(
        commandBuffer: MTLCommandBuffer,
        previousState: MTLTexture,
        newState: MTLTexture
    ) {
        gol.encode(
            commandBuffer: commandBuffer,
            sourceTexture: previousState,
            destinationTexture: newState
        )
    }
}

final class ConvayViewController: GameOfLifeViewController {
    override var setup: GameOfLifeViewController.Setup {
        .init(gridType: .moore1, kind: .convay)
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 32, height: 32)
    }
    
    override var shouldPreserveSquareCells: Bool {
        true
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        cgContext.fill(shape: .rPentomino)
    }
}

final class SeedsViewController: GameOfLifeViewController {
    override var setup: GameOfLifeViewController.Setup {
        .init(gridType: .moore1, kind: .seeds)
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 1024, height: 1024)
    }
    
    override var shouldPreserveSquareCells: Bool {
        true
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        cgContext.fill(shape: .glider)
    }
}

final class BriansBrainViewController: GameOfLifeViewController {
    override var setup: GameOfLifeViewController.Setup {
        .init(gridType: .moore1, kind: .brianBrain)
    }
    
    override var colors: [CGColor] {
        [
            .white, .blue, .black
        ]
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 4096, height: 4096)
    }
    
    override var shouldPreserveSquareCells: Bool {
        true
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        cgContext.fill(shape: .randomShape(width: 111, height: 111))
    }
}

final class WireWorldViewController: GameOfLifeViewController {
    override var setup: GameOfLifeViewController.Setup {
        .init(gridType: .moore1, kind: .wireWorld)
    }
    
    private lazy var copyTexture = CopyTexture(context: context)
    private lazy var mapTexture: MTLTexture = {
        let loader = MTKTextureLoader(device: context.device)
        let url = Bundle.main.url(forResource: "wireworld", withExtension: "gif")!
        return try! loader.newTexture(URL: url)
    }()
    
    override var arenaDimensions: CGSize {
        .init(width: mapTexture.width, height: mapTexture.height)
    }
    
    override var colors: [CGColor] {
        [
            .blue, .red, .yellow, .black
        ]
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let commandBuffer = context.commandQueue.makeCommandBuffer()!
        copyTexture.encode(commandBuffer: commandBuffer, sourceTexture: mapTexture, destinationTexture: self.texture)
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
    }
}
