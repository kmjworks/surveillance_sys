#include <ros/ros.h>
#include <functional>

class EventLoopTimeKeeper {
    public:
        explicit EventLoopTimeKeeper(double frequency) : period(1.0 / frequency), lastExecutionTime(ros::Time::now()) {}

        bool shouldRun() {
            ros::Time currentTime = ros::Time::now();
            double elapsedTime = (currentTime - lastExecutionTime).toSec();

            if(elapsedTime >= period) {
                lastExecutionTime = currentTime;
                return true;
            }
            return false;
        }

        template <typename Func>
        bool run(Func&& func) {
            if (shouldRun()) {
                func();
                return true;
            }
            return false;
        }

    private:
        double period; 
        ros::Time lastExecutionTime;
};