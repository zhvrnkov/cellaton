//
//  LongtonAntViewController.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/19/22.
//

import Foundation
import simd
import Metal

final class LangtonAntViewController: GameOfLifeViewController {
    
    struct Ant {
        var location: vector_long2
        var angle: Float
        var rule: AntRule = .RL
        
        var speed: Int = 1
        
        mutating func step(texture: MTLTexture) {
            if location.x < 0 {
                location.x += texture.width
            }
            if location.y < 0 {
                location.y += texture.height
            }
            location.x %= texture.width
            location.y %= texture.height

            let antPixel = texture.getRGBA(x: location.x, y: location.y)
            let (newColor, deltaAngle) = rule[antPixel]!
            texture.write(rgba: newColor, x: location.x, y: location.y)
            angle += deltaAngle

            let antDirection = vector_long2(x: .init(cos(angle).rounded()), y: .init(sin(angle).rounded()))
            location = location &+ antDirection
        }
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 512, height: 512)
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
    
    private lazy var ants: [Ant] = {
        var output = [Ant]()
//        let rule = AntRule.rules.randomElement()!
        let rule = AntRule.LLRR
        output.append(.init(location: .zero, angle: 0))
        output.append(.init(location: .init(x: texture.width - 1, y: texture.height - 1), angle: 0))
        output.append(.init(location: .init(x: texture.width / 2 - 1, y: texture.height / 2 - 1), angle: 0))
        
        for index in output.indices {
            output[index].rule = rule
            output[index].speed = 100
        }
        return output
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        for _ in 0..<0 {
            for index in ants.indices {
                ants[index].step(texture: self.texture)
            }
        }
    }
    
    override func encode(commandBuffer: MTLCommandBuffer, previousState: MTLTexture, newState: MTLTexture) {
        for index in ants.indices {
            for _ in 0..<ants[index].speed {
                ants[index].step(texture: newState)
            }
        }
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
    
    static var rules: [Self] {
        [RL, RRL, RLR, LLRR, LRRRRRLLR, LLRRRLRLRLLR, RRLLLRLLLRRR]
    }
    
    static var RL: Self = .rule([R, L])
    
    static var RRL: Self {
        [
            white: (black, R),
            black: (red, L),
            red: (white, R),
        ]
    }
    
    static var RLR: Self {
        rule([R, L, R])
    }
    
    static var LLRR: Self {
        [
            white: (purple, R),
            purple: (bluegre, R),
            bluegre: (black, L),
            black: (white, L),
        ]
    }
    
    static var LRRRRRLLR: Self {
        rule([L,R,R,R,R,R,L,L,R])
    }
    
    static var LLRRRLRLRLLR: Self {
        rule([L,L,R,R,R,L,R,L,R,L,L,R])
    }
    
    static var RRLLLRLLLRRR: Self {
        rule([R,R,L,L,L,R,L,L,L,R,R,R])
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
