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
    
    var data: (Grid, GOLRule, StateToDeltaTable) {
        setup.data
    }
    
    private lazy var gol: GOLKernel = {
        let kernel = GOLKernel(context: context)
        let (grid, rules, table) = data
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

final class LongtonAntViewController: GameOfLifeViewController {
    override var colors: [CGColor] {
        [.white, .red, .black]
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 256, height: 256)
    }
    
    override var inProgressFPS: Int {
       60
    }
    
    override var shouldPreserveSquareCells: Bool {
        false
    }
    
    override var data: (Grid, GOLRule, StateToDeltaTable) {
        let grid: [[Int32]] = [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
        var rule = GOLRule.golRule(gridLength: grid.length)
        rule[0b000] = GOLRule.rule(states: [], defaultState: 0b000, count: grid.length)
        rule[0b111] = GOLRule.rule(states: [(1, 0b100)], defaultState: 0b111, count: grid.length)
        rule[0b100] = GOLRule.rule(states: [], defaultState: 0b000, count: grid.length)
        
        var table: StateToDeltaTable = .init(repeating: 0, count: 1<<3)
        table[0b100] = 1
        return (grid, rule, table)
    }
    
    private lazy var antLocation = vector_long2(x: cgContext.width / 2 - 1, y: cgContext.height / 2 - 1)
    private lazy var antAngle: Float = .pi
    
    
    private lazy var rule: AntRule = .rule([R,R,L,L,L,R,L,L,L,R,R,R])
//    private lazy var rule: AntRule = .rule([R, R, L, L])
    
    var index: UInt = 0
    
    override func viewDidLoad() {
        super.viewDidLoad()
        for i in 0..<0 {
            step()
        }
    }
    
    override func encode(commandBuffer: MTLCommandBuffer, previousState: MTLTexture, newState: MTLTexture) {
        for _ in 0..<10 {
            step(texture: newState)
        }
    }
    
    private func step(texture: MTLTexture? = nil) {
        let texture = texture ?? self.texture!
        if antLocation.x < 0 {
            antLocation.x += texture.width
        }
        if antLocation.y < 0 {
            antLocation.y += texture.height
        }
        antLocation.x %= texture.width
        antLocation.y %= texture.height

        let antPixel = texture.getRGBA(x: antLocation.x, y: antLocation.y)
        let (newColor, deltaAngle) = rule[antPixel]!
        texture.write(rgba: newColor, x: antLocation.x, y: antLocation.y)
        antAngle += deltaAngle

        let antDirection = vector_long2(x: .init(cos(antAngle).rounded()), y: .init(sin(antAngle).rounded()))
        antLocation = antLocation &+ antDirection
    }
}

typealias AntRule = [UInt32: (color: UInt32, angle: Float)]

let white: UInt32 = 0xffffffff
let black: UInt32 = 0xff000000
let red: UInt32 = 0xff0000ff
let green: UInt32 = 0xff00ff00
let blue: UInt32 = 0xffff0000
let yellow: UInt32 = 0xff00ffff
let purple: UInt32 = 0xffff00ff
let bluegre: UInt32 = 0xffffff00

let colors = [
    red,
    green,
    blue,
    yellow,
    purple,
    bluegre,
    0xffaaaaaa,
    0xffffaabb,
    0xff0bfeff,
    0xff1212ff,
    0xffaafffe,
    0xffba1f0e,
    0xff1f1823,
]

let R = -Float.pi / 2
let L = Float.pi / 2
let N = Float(0)

extension AntRule {
    static var RL: Self = .rule([R, L])
    
    static var RRL: Self {
        [
            white: (black, R),
            black: (red, L),
            red: (white, R),
        ]
    }
    
    static var LLRR: Self {
        [
            white: (purple, R),
            purple: (bluegre, R),
            bluegre: (black, L),
            black: (white, L),
        ]
    }
    
    static func rule(_ angles: [Float]) -> Self {
        var output: [(UInt32, (UInt32, Float))] = []
        let colors = [black] + colors.shuffled()[0..<angles.count - 1]
        
        for (index, angle) in angles.enumerated() {
            output.append((colors[index], (colors[(index + 1) % colors.count], angle)))
        }
        
        return .init(uniqueKeysWithValues: output)
    }
    
}
