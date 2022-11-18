//
//  AppDelegate.swift
//  game-of-life
//
//  Created by Zhavoronkov Vlad on 11/15/22.
//

import UIKit

@main
class AppDelegate: UIResponder, UIApplicationDelegate {

    lazy var window: UIWindow? = .init(frame: UIScreen.main.bounds)

    func application(
        _ application: UIApplication,
        didFinishLaunchingWithOptions launchOptions: [UIApplication.LaunchOptionsKey: Any]?
    ) -> Bool {
        let menu = MenuViewController(controllersType: [
            Rule110ViewController.self,
            Rule90ViewController.self,
            Rule30ViewController.self,
            ConvayViewController.self,
            SeedsViewController.self,
            BriansBrainViewController.self,
            WireWorldViewController.self
        ])
        let navigationController = UINavigationController(rootViewController: menu)
        navigationController.pushViewController(WireWorldViewController(), animated: false)
        window?.rootViewController = navigationController
        window?.makeKeyAndVisible()
        return true
    }
}

