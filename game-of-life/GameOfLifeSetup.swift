//
//  GameOfLifeSetup.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/18/22.
//

import Foundation

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
            case wireWorld
            
            func data(gridLength: Int) -> GOLRule {
                switch self {
                case .convay:
                    return .convay(gridLength: gridLength)
                case .seeds:
                    return .seeds(gridLength: gridLength)
                case .brianBrain:
                    return .brianBrain(gridLength: gridLength)
                case .wireWorld:
                    return .wireWorld(gridLength: gridLength)
                }
            }
            
            func stateToDeltaTable() -> StateToDeltaTable {
                switch self {
                case .convay,
                        .brianBrain,
                        .seeds:
                    return .liveDeadStateToDelta
                case .wireWorld:
                    return .wireWorldStateToDelta
                }
            }
        }
        
        let gridType: GridType
        let kind: Kind
        
        var data: (Grid, GOLRule, StateToDeltaTable) {
            let grid = gridType.data
            let rules = kind.data(gridLength: grid.length)
            let table = kind.stateToDeltaTable()
            return (grid, rules, table)
        }
    }
}


