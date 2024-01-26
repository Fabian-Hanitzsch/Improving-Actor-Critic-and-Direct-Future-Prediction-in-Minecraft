if [ ! -f /tmp/.X99-lock ]; then
    Xvfb :99 -screen 0 1024x768x24+32 +extension GLX +render </dev/null &
    echo "Waiting for Xvfb to be ready..."
  	while ! xdpyinfo -display ":99" > /dev/null 2>&1; do
  		sleep 0.1
    done
else
    echo "INFO: $(date) - X Server already running" 1>&2
fi

export DISPLAY=":99"
echo "Xvfb is ready and running"
node --stack-trace-limit=1000 index.js