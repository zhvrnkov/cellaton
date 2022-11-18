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
        60
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
        .init(width: 4096, height: 4096)
    }
    
    override var shouldPreserveSquareCells: Bool {
        true
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        cgContext.setFillColor(.init(gray: 1, alpha: 1))
        var origin = CGPoint(x: cgContext.width / 2 - 1, y: cgContext.height / 2 - 1)
        
//        var grid: [[Int]] = [
//            [0, 0, 0, 0, 0],
//            [0, 1, 1, 0, 0],
//            [0, 1, 1, 0, 0],
//            [0, 0, 0, 0, 0],
//            [0, 0, 0, 0, 0],
//        ]
//        cgContext.fill(shape: grid, color: CGColor(gray: 1, alpha: 1))
//        grid = [
//            [0, 0, 1, 0, 0],
//            [1, 0, 0, 0, 0],
//            [0, 0, 0, 1, 0],
//            [0, 1, 0, 0, 0],
//            [0, 0, 0, 0, 0],
//        ]
//        cgContext.fill(shape: grid, color: CGColor(red: 0, green: 0, blue: 1, alpha: 1))
        let count = 111
        cgContext.fill(shape: .randomShape(width: count, height: count))
//        cgContext.fill(shape: .pants)
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


