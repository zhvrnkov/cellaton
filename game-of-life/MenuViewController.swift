import Foundation
import UIKit

open class MenuViewController: UIViewController {
    
    open lazy var stackView: UIStackView = {
        let subviews = controllersType.map { vcType in
            makeButton(type: vcType)
        }
        let view = UIStackView(arrangedSubviews: subviews)
        view.axis = .vertical
        view.distribution = .fillProportionally
        return view
    }()
    
    public let controllersType: [UIViewController.Type]
    
    public init(controllersType: [UIViewController.Type]) {
        self.controllersType = controllersType
        super.init(nibName: nil, bundle: nil)
    }
    
    required public init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    open override func viewDidLoad() {
        super.viewDidLoad()
        view.addSubview(stackView)
    }
    
    open override func viewDidLayoutSubviews() {
        super.viewDidLayoutSubviews()
        stackView.frame = view.safeAreaBounds
    }
    
    open func makeButton<VC: UIViewController>(type: VC.Type) -> UIButton {
        let button = Button(type: .system) { [weak self] in
            guard let self else {
                return
            }
            let viewController = type.init()
            self.navigationController?.pushViewController(viewController, animated: true)
        }
        button.setTitle(String(describing: type), for: .normal)
        return button
    }
}

final class Button: UIButton {
    
    private var action: (() -> Void)?
    
    convenience init(type: UIButton.ButtonType, action: (() -> Void)?) {
        self.init(type: type)
        self.action = action
        
        self.addTarget(self, action: #selector(pressed), for: .touchUpInside)
    }
    
    @objc private func pressed() {
        action?()
    }
}
