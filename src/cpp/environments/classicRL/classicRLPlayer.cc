#include <GL/gl.h>
#include <GL/glut.h>
#include <cartPole.h>
#include <acrobot.h>
#include <cartCentering.h>
#include <pendulum.h>
#include <mountainCar.h>

using namespace std;
typedef double stateType;

TaskEnv *_game;
int _action;
int _millis;
double _reward;
double _width = 1000;
double _height = 1000;
mt19937 _rng;

void display_function(void)
{
   _game->display_function(0, _action, 0.0);
}

void catchKey(int key, int x, int y)
{
   (void)x;
   (void)y;
   if (key == GLUT_KEY_LEFT)
      _action = 0;
   else if (key == GLUT_KEY_RIGHT)
      _action = 2;
   else if (key == GLUT_KEY_DOWN)
      _action = 1;
}

void timer_function(int value)
{
   (void)value; //unused parameter
   if (_game->terminal()){
      cout << " reward " << _reward << endl;
      exit(0);
   }

   ////optimal cart centering controller
   //vector <double> s = _game->getStateVec(false);
   //if (-1.0*s[0] > s[1]*abs(s[1])) _action = 2; //1.0;
   //else _action = 0;//-1.0;
 
   _reward += _game->update(_action, 0.0, _rng);

   glutPostRedisplay();
   glutTimerFunc(_millis, timer_function , 0);
}

int main ( int argc , char ** argv)
{
   if (atoi(argv[1]) == 1)
      _game = new cartPole();
   else if (atoi(argv[1]) == 2)
      _game = new acrobot();
   else if (atoi(argv[1]) == 3)
      _game = new cartCentering();
   else if (atoi(argv[1]) == 4)
      _game = new pendulum();
   else if (atoi(argv[1]) == 5)
      _game = new mountainCar();
   else
      _game = new cartPole();

   _rng.seed(atoi(argv[2]));
   _millis = atoi(argv[3]);

   _action = 1;
   _game->reset(_rng);

   glutInit( &argc , argv);
   glutInitDisplayMode(GLUT_SINGLE |GLUT_RGB);
   glutInitWindowSize(_width, _height);
   glutInitWindowPosition(100,100);
   glutCreateWindow("Cartpole");

   glScalef(0.5, 0.5, 0.0);

   glutDisplayFunc ( display_function);
   glutTimerFunc (1, timer_function,0);
   glutSpecialFunc(catchKey);
   glutMainLoop();
}
