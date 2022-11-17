//
//  GameOfLifeViewController.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/17/22.
//

import Foundation
import Metal
import CoreGraphics

final class GameOfLifeViewController: CommonViewController {
    override var inProgressFPS: Int {
        15
    }
    
    private let setup: Setup = {
        return .init(
            gridType: .moore1,
            kind: .brianBrain
        )
    }()
    private lazy var gol: GOLKernel = {
        let kernel = GOLKernel(context: context)
        let (grid, rules) = setup.data
        kernel.setup(grid: grid, rules: rules)
        return kernel
    }()
    
    override var arenaDimensions: CGSize {
        .init(width: 1024, height: 1024)
    }
    
    override var shouldPreserveSquareCells: Bool {
        true
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        cgContext.setFillColor(.init(gray: 1, alpha: 1))
        var origin = CGPoint(x: cgContext.width / 2 - 1, y: cgContext.height / 2 - 1)
        
//        var grid: [[Int]] = [
//            [1, 0, 0, 0, 1],
//            [0, 1, 1, 0, 0],
//            [0, 1, 1, 1, 0],
//            [0, 0, 1, 1, 0],
//            [1, 0, 0, 0, 1],
//        ]
//        fill(grid: grid)
//        grid = [
//            [0, 1, 0, 0, 0],
//            [1, 0, 0, 1, 1],
//            [1, 0, 0, 0, 0],
//            [0, 0, 1, 0, 0],
//            [0, 1, 0, 0, 0],
//        ]
//        fill(grid: grid, color: CGColor(red: 0, green: 0, blue: 1, alpha: 1))
        let count = cgContext.height
        var grid: [[Int]] = (0..<count).map { _ in
            (0..<count).map { _ in
                Bool.random() ? 1 : 0
            }
        }
        fill(grid: grid)
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
    
    private func fill(grid: [[Int]], center: CGPoint? = nil, color: CGColor = CGColor(gray: 1, alpha: 1)) {
        let center = center ?? CGPoint(x: cgContext.width / 2 - 1, y: cgContext.height / 2 - 1)
        let rangeY = grid.count / 2
        let rangeX = grid[0].count / 2
        
        for y in -rangeY...rangeY {
            let lookY = y + rangeY
            for x in -rangeX...rangeX {
                let lookX = x + rangeX
                let point = CGPoint(x: Int(center.x) + x, y: Int(center.y) + y)
//                let color = grid[lookY][lookX] == 0 ? CGColor(gray: 0, alpha: 1) : color
                if (grid[lookY][lookX] == 0) == false {
                    cgContext.setFillColor(color)
                    cgContext.fillPixel(at: point)
                }
            }
        }
    }
}

extension GameOfLifeViewController {
    struct Setup {
        enum GridType {
            case moore1
            case moore2
            case vonNeumann1
            case vonNeumann2
            
            var data: Grid {
                switch self {
                case .moore1:
                    return .moore1
                case .moore2:
                    return .moore2
                case .vonNeumann1:
                    return .vonNeumann1
                case .vonNeumann2:
                    return .vonNeumann2
                }
            }
        }
        
        enum Kind {
            case convay
            case seeds
            case brianBrain
            
            func data(gridLength: Int) -> GOLRule {
                switch self {
                case .convay:
                    return .convay(gridLength: gridLength)
                case .seeds:
                    return .seeds(gridLength: gridLength)
                case .brianBrain:
                    return .brianBrain(gridLength: gridLength)
                }
            }
        }
        
        let gridType: GridType
        let kind: Kind
        
        var data: (Grid, GOLRule) {
            let grid = gridType.data
            let rules = kind.data(gridLength: grid.length)
            return (grid, rules)
        }
    }
}

extension [[Int]] {
    static var glider: Self {
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    }
    
    static var rPentomino: Self {
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0],
        ]
    }
    
    static var todd: Self {
        [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0],
        ]
    }

    static var square: Self {
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    }
    
    static var pants: Self {
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    }
    
    static var threeSquares: Self {
        [
            [1, 1, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 1, 1],
        ]
    }


}
