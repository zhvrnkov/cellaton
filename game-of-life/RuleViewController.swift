//
//  Rule110ViewController.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/17/22.
//

import Foundation
import CoreGraphics
import Metal

class RuleViewController: CommonViewController {
    
    enum InitialState {
        case mid
        case left
        case right
        case random
        
        func initialize(context: CGContext) {
            switch self {
            case .mid:
                let x = context.width / 2 - 1
                context.setFillColor(CGColor(gray: 1, alpha: 1))
                context.fillPixel(at: .init(x: x, y: 0))
            case .left:
                let x = 0
                context.setFillColor(CGColor(gray: 1, alpha: 1))
                context.fillPixel(at: .init(x: x, y: 0))
            case .right:
                let x = context.width - 1
                context.setFillColor(CGColor(gray: 1, alpha: 1))
                context.fillPixel(at: .init(x: x, y: 0))
            case .random:
                for x in 0..<context.width {
                    context.setFillColor(CGColor(gray: Bool.random() ? 1 : 0, alpha: 1))
                    context.fillPixel(at: .init(x: x, y: 0))
                }
            }
        }
    }
    
    var rule: Rule {
        []
    }
    var initialState: InitialState {
        .mid
    }
    
    var instantRendering: Bool {
        false
    }
    
    private lazy var ruleKernel: RuleKernel = {
        let kernel = RuleKernel(context: context)
        kernel.rule = rule
        return kernel
    }()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        initialState.initialize(context: cgContext)
    }
    
    override func encode(
        commandBuffer: MTLCommandBuffer,
        previousState: MTLTexture,
        newState: MTLTexture
    ) {
        if instantRendering {
            while ruleKernel.offset.y < newState.height {
                ruleKernel.offset.y += 1
                ruleKernel.encode(commandBuffer: commandBuffer, destinationTexture: newState)
            }
        }
        else {
            ruleKernel.offset.y += 1
            ruleKernel.encode(commandBuffer: commandBuffer, destinationTexture: newState)
        }
    }
}

final class Rule110ViewController: RuleViewController {
    override var rule: Rule {
        .rule110
    }
    
    override var initialState: RuleViewController.InitialState {
        .right
    }
    
    override var shouldPreserveSquareCells: Bool {
        return false
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 1024, height: 1024)
    }
    
    override var instantRendering: Bool {
        true
    }
}

final class Rule90ViewController: RuleViewController {
    override var rule: Rule {
        .rule90
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 2048, height: 2048)
    }
    
    override var instantRendering: Bool {
        true
    }
}

final class Rule30ViewController: RuleViewController {
    override var rule: Rule {
        .rule30
    }
    
    override var arenaDimensions: CGSize {
        .init(width: 2048, height: 2048)
    }
    
    override var instantRendering: Bool {
        false
    }
}
