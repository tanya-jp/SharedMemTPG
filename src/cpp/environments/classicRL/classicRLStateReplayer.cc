#include <GL/gl.h>
#include <GL/glut.h>
#include <cartPole.h>
#include <acrobot.h>
#include <cartCentering.h>
#include <pendulum.h>
#include <mountainCar.h>
#include <unistd.h>

using namespace std;
typedef double stateType;

TaskEnv *_game;
double _actionC;
int _actionD;
int _millis;
double _reward;
double _width = 1000;
double _height = 1000;
mt19937 _rng;
vector < vector< stateType >> stateSequence; //state, act, 
size_t step;

void display_function(void)
{
   _game->display_function(0, _actionD, _actionC);
}

//void catchKey(int key, int x, int y)
//{
//   (void)x;
//   (void)y;
//   if (key == GLUT_KEY_LEFT)
//      _action = 0;
//   else if (key == GLUT_KEY_RIGHT)
//      _action = 2;
//   else if (key == GLUT_KEY_DOWN)
//      _action = 1;
//}

void timer_function(size_t step)
{

   if (stateSequence[step][0] == 0 && stateSequence[step][1] == 0)
      _game = new cartPole();
   else if (stateSequence[step][0] == 1 && stateSequence[step][1] == 0)
      _game = new cartCentering();
   else if (stateSequence[step][0] == 2 && stateSequence[step][1] == 0)
      _game = new mountainCar();

   //_reward += _game->update(_action, 0.0, _rng);
   _game->setStep(stateSequence[step][1]);
   _game->setStateVar(0, stateSequence[step][2]);
   _game->setStateVar(1, stateSequence[step][3]);
   _game->setStateVar(2, stateSequence[step][4]);
   _game->setStateVar(3, stateSequence[step][5]);
   _actionD = stateSequence[step][6] <= 0 ? 0 : 2;
   _actionC = stateSequence[step][6];
   
   
   //glutPostRedisplay();
   if (stateSequence[step][1] == 0 || (step < stateSequence.size()-1 && stateSequence[step+1][1] == 0))
      sleep(3);
   //glutTimerFunc(_millis, timer_function , 0);
   //if (step == stateSequence.size()-1){
   //   sleep(3);
   //   exit(0);
   //}
}

int main ( int argc , char ** argv)
{
   _actionD = _actionC = step = 0;
   //if (atoi(argv[1]) == 1)
   //   _game = new cartPole();
   //else if (atoi(argv[1]) == 2)
   //   _game = new acrobot();
   //else if (atoi(argv[1]) == 3)
   //   _game = new cartCentering();
   //else if (atoi(argv[1]) == 4)
   //   _game = new pendulum();
   //else if (atoi(argv[1]) == 5)
   //   _game = new mountainCar();
   //else
      _game = new cartPole();

   _rng.seed(0);
   _millis = atoi(argv[2]);

   //_game->reset(_rng);

   //read state sequence ///////////////////////////////////////////////////
   ifstream infile("stateSeq.rslt");
   string line;
   vector < string > outcomeFields;
   vector <stateType> stateline;
   while (getline(infile, line))
   {
      outcomeFields.clear();
      stateline.clear();
      splitString(line,' ',outcomeFields);
      stateline.push_back(atof(outcomeFields[2].c_str()));
      for (size_t i = 4; i < outcomeFields.size(); i++)
         stateline.push_back(atof(outcomeFields[i].c_str()));
      stateSequence.push_back(stateline);
   }

   glutInit( &argc , argv);
   glutInitDisplayMode(GLUT_SINGLE |GLUT_RGB);
   glutInitWindowSize(_width, _height);
   glutInitWindowPosition(100,100);
   glutCreateWindow("classicRL");

   glScalef(0.5, 0.5, 0.0);

   //glutDisplayFunc ( display_function);
   //glutTimerFunc (1, timer_function,0);
   //glutMainLoop();
   for (size_t step = 0; step < stateSequence.size(); step++){
      if (stateSequence[step][0] == 0 && stateSequence[step][1] == 0)
      _game = new cartPole();
   else if (stateSequence[step][0] == 1 && stateSequence[step][1] == 0)
      _game = new cartCentering(); 
   else if (stateSequence[step][0] == 2 && stateSequence[step][1] == 0)
      _game = new mountainCar();

   //_reward += _game->update(_action, 0.0, _rng);
   _game->setStep(stateSequence[step][1]);
   _game->setStateVar(0, stateSequence[step][2]);
   _game->setStateVar(1, stateSequence[step][3]);
   _game->setStateVar(2, stateSequence[step][4]);
   _game->setStateVar(3, stateSequence[step][5]);
   _actionD = stateSequence[step][6] <= 0 ? 0 : 2;
   _actionC = stateSequence[step][6];

   _game->display_function(0, _actionD, _actionC);

   //glutPostRedisplay();
   usleep(_millis * 1000);
   if (step == stateSequence.size()-1 || stateSequence[step][1] == 0 || (step < stateSequence.size()-1 && stateSequence[step+1][1] == 0))
      sleep(3);
   }
}
