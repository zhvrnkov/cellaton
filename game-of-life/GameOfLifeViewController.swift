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
        30
    }
    
    private lazy var gol = GOLKernel(context: context)
    
    override var arenaDimensions: CGSize {
        .init(width: 512, height: 512)
    }
    
    override var shouldPreserveSquareCells: Bool {
        true
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        cgContext.setFillColor(.init(gray: 1, alpha: 1))
        var origin = CGPoint(x: cgContext.width / 2 - 1, y: cgContext.height / 2 - 1)
        
        let grid: [[Int]] = [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 0],
        ]
        
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
    
    private func fill(grid: [[Int]], center: CGPoint? = nil) {
        let center = center ?? CGPoint(x: cgContext.width / 2 - 1, y: cgContext.height / 2 - 1)
        let rangeY = grid.count / 2
        let rangeX = grid[0].count / 2
        
        for y in -rangeY...rangeY {
            let lookY = y + rangeY
            for x in -rangeX...rangeX {
                let lookX = x + rangeX
                let point = CGPoint(x: Int(center.x) + x, y: Int(center.y) + y)
                let color = CGColor(gray: grid[lookY][lookX] == 0 ? 0 : 1, alpha: 1)
                cgContext.setFillColor(color)
                cgContext.fillPixel(at: point)
            }
        }
    }
}
