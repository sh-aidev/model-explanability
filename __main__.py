import sys
from src.image_exp import ImageExplanability

def main():
    image_exp = ImageExplanability()
    image_exp.run_explanability_img()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('Keyboard interrupt detected. Exiting...')
        sys.exit(0)
    except Exception as e:
        print(f'Exited with error: {e}')
        sys.exit(0)

