//
//  GOLRule.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/18/22.
//

import Foundation

// State = ColorComponentsSum = 0, 1, 2, 3
// NeibghorsCount = Int
// Color = f4 = vector_float4
typealias GOLRule = [[f4]] // [State: [NeibghorsCount: Color]]

extension f4 {
    init(int: Int) {
        let r = (int >> 2) & 1
        let g = (int >> 1) & 1
        let b = (int >> 0) &  1
        
        self = f4(Float(r), Float(g), Float(b))
    }
}

typealias StateToDeltaTable = [Int32]
extension StateToDeltaTable {
    static var liveDeadStateToDelta: Self {
        let liveState = 0b111
        let deadState = 0b000
        var output = Self(repeating: 0, count: 1<<3)
        output[liveState] = 1
        output[deadState] = 0
        return output
    }
    
    static var wireWorldStateToDelta: Self {
        let empty =     0b000
        let eHead =     0b001
        let eTail =     0b100
        let conductor = 0b110
        var output = Self(repeating: 0, count: 1<<3)
        output[empty] = 0
        output[eHead] = 1
        output[eTail] = 0
        output[conductor] = 0
        return output
    }
}

extension GOLRule {
    
    static func convay(gridLength: Int) -> Self {
        let liveState = 0b111
        let deadState = 0b000
        var output = golRule(gridLength: gridLength)
#warning("max number in index can be 9 (if surrounded by lives and live itself, but length of array is 9 (so max index is 8). Don't detemine rule array length in shader by gridDim + 1, but pass separate parameter to it and rule length should be gridLength + 1")
        let deadRule = rule(
            states: [(3, liveState)],
            defaultState: deadState,
            count: gridLength
        )
        let liveRule = rule(
            states: [(2, liveState), (3, liveState)],
            defaultState: deadState,
            count: gridLength
        )

        output[liveState] = liveRule
        output[deadState] = deadRule
        
        return output
    }
    
    static func seeds(gridLength: Int) -> Self {
        let liveState = 0b111
        let deadState = 0b000
        var output = golRule(gridLength: gridLength)
        let liveRule = rule(
            states: [],
            defaultState: deadState,
            count: gridLength)
        let deadRule = rule(
            states: [(2, liveState)],
            defaultState: deadState,
            count: gridLength
        )
        
        output[liveState] = liveRule
        output[deadState] = deadRule
        
        return output
    }
    
    static func brianBrain(gridLength: Int) -> Self {
        let liveState = 0b111
        let dyingState = 0b001
        let deadState = 0b000
        var output = golRule(gridLength: gridLength)
        
        let liveRule = rule(states: [], defaultState: dyingState, count: gridLength)
        let dyingRule = rule(states: [], defaultState: deadState, count: gridLength)
        let deadRule = rule(states: [(2, liveState)], defaultState: deadState, count: gridLength)
        
        output[deadState] = deadRule
        output[dyingState] = dyingRule
        output[liveState] = liveRule
        
        return output
    }
    
    static func wireWorld(gridLength: Int) -> Self {
        let empty =     0b000
        let eHead =     0b001
        let eTail =     0b100
        let conductor = 0b110
        var output = golRule(gridLength: gridLength)
        
        output[empty] = rule(states: [], defaultState: empty, count: gridLength)
        output[eHead] = rule(states: [], defaultState: eTail, count: gridLength)
        output[eTail] = rule(states: [], defaultState: conductor, count: gridLength)
        output[conductor] = rule(
            states: [(1, eHead), (2, eHead)],
            defaultState: conductor,
            count: gridLength
        )
        
        return output
    }
    
    static func golRule(gridLength: Int) -> Self {
        let statesCount = 0b111 + 1
        return [[f4]](repeating: rule([], gridLength), count: statesCount)
    }
}

