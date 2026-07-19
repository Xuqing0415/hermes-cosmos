import importlib
import os
from typing import Dict, Optional, List, Type
from .interfaces import DomainPlugin


class PluginManager:
    def __init__(self, plugins_dir: Optional[str] = None):
        self.plugins_dir = plugins_dir or self._find_plugins_dir()
        self._plugins: Dict[str, DomainPlugin] = {}
        self._plugin_classes: Dict[str, Type[DomainPlugin]] = {}
    
    def _find_plugins_dir(self) -> str:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        plugins_dir = os.path.join(parent_dir, "plugins")
        return plugins_dir
    
    def discover_plugins(self) -> List[str]:
        plugins = []
        
        if not os.path.exists(self.plugins_dir):
            return plugins
        
        for item in os.listdir(self.plugins_dir):
            item_path = os.path.join(self.plugins_dir, item)
            if os.path.isdir(item_path):
                init_file = os.path.join(item_path, "__init__.py")
                if os.path.exists(init_file):
                    plugins.append(item)
        
        return plugins
    
    def load_plugin(self, plugin_name: str) -> Optional[DomainPlugin]:
        if plugin_name in self._plugins:
            return self._plugins[plugin_name]
        
        try:
            module_path = f"hermes.plugins.{plugin_name}"
            module = importlib.import_module(module_path)
            
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if isinstance(attr, type) and issubclass(attr, DomainPlugin) and attr != DomainPlugin:
                    plugin_instance = attr()
                    self._plugins[plugin_name] = plugin_instance
                    self._plugin_classes[plugin_name] = attr
                    return plugin_instance
        
        except ImportError:
            pass
        except Exception:
            pass
        
        return None
    
    def get_plugin(self, domain_name: str) -> Optional[DomainPlugin]:
        if domain_name in self._plugins:
            return self._plugins[domain_name]
        
        for plugin_name in self.discover_plugins():
            plugin = self.load_plugin(plugin_name)
            if plugin and plugin.domain_name == domain_name:
                return plugin
        
        return None
    
    def list_plugins(self) -> List[Dict[str, str]]:
        plugins = []
        for plugin_name in self.discover_plugins():
            plugin = self.load_plugin(plugin_name)
            if plugin:
                plugins.append({
                    "name": plugin_name,
                    "domain": plugin.domain_name,
                    "description": plugin.description
                })
        return plugins
    
    def unload_plugin(self, plugin_name: str):
        if plugin_name in self._plugins:
            del self._plugins[plugin_name]
        if plugin_name in self._plugin_classes:
            del self._plugin_classes[plugin_name]