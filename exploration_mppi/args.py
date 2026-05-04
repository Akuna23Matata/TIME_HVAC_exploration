"""
Argument parser for MPPI Controller experiments
"""
import argparse

def create_parser():
    """Create and return the argument parser"""
    parser = argparse.ArgumentParser(description='MPPI Controller for HVAC Control')
    
    # Add arguments
    parser.add_argument('--timestep', 
                       type=int, 
                       default=4,
                       help='time step per hour')
    
    parser.add_argument('--environment', 
                       type=str, 
                       default='Eplus-5zone-hot-discrete-v1',
                       help='Environment name')
    
    parser.add_argument('--weather', 
                       type=str, 
                       default=None,
                       help='Weather file name (optional)')
    
    parser.add_argument('--winter', 
                       action='store_true',
                       help='Enable winter mode (heating only)')
    parser.add_argument('--summer',
                        action='store_true',
                        help='enable summer mode (cooling only)')
    parser.add_argument("--traindays",
                        type=int,
                        default=14,
                        help="days of data collection")
    
    return parser

def parse_args():
    """Parse and return command line arguments"""
    parser = create_parser()
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Use the arguments
    # print(f"Horizon: {args.horizon}")
    print(f"Environment: {args.environment}")
    print(f"Verbose: {args.summer}")