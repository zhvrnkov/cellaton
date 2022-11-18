//
//  Shapes.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/18/22.
//

import Foundation
import CoreGraphics

typealias Shape = [[Int]]

extension Shape {
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
    
    static func randomShape(width: Int, height: Int) -> Self {
        (0..<height).map { _ in
            (0..<width).map { _ in
                Bool.random() ? 1 : 0
            }
        }
    }
    
    static func solidShape(width: Int, height: Int) -> Self {
        (0..<height).map { _ in
            (0..<width).map { _ in
                1
            }
        }
    }
}

extension CGContext {
    func fill(
        shape: Shape,
        center: CGPoint? = nil,
        color: CGColor = CGColor(gray: 1, alpha: 1)
    ) {
        
        let center = center ?? CGPoint(x: width / 2 - 1, y: height / 2 - 1)
        let rangeY = shape.count / 2
        let rangeX = shape[0].count / 2
        
        for y in -rangeY...rangeY {
            let lookY = y + rangeY
            for x in -rangeX...rangeX {
                let lookX = x + rangeX
                let point = CGPoint(x: Int(center.x) + x, y: Int(center.y) + y)
                if (shape[lookY][lookX] == 0) == false {
                    setFillColor(color)
                    fillPixel(at: point)
                }
            }
        }
    }
}
