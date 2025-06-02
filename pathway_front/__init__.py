"""Paquete *pathway_front* – capa de presentación de la aplicación Pathway.

Este paquete agrupa la definición de *componentes* de la interfaz de usuario
(construidos con Reflex), el *estado* reactivo que comparten y la *guía de
estilos* centralizada.  Su objetivo es separar claramente la lógica de
presentación del resto de la aplicación para mantener:

1. Alta cohesión: cada módulo (``state``, ``style`` y ``pathway_front``)
   aborda un único cometido.
2. Bajo acoplamiento: los componentes dependen solo de las API públicas
   expuestas por ``state`` y de los tokens de ``style`` – nunca de detalles
   internos del *back-end*.

Los módulos son importados por Reflex en *tiempo de compilación* para generar
la aplicación React / Next.js que finalmente se sirve al navegador.  Mantener
un límite limpio entre *front-end* y *back-end* reduce errores de
serialización y facilita las pruebas unitarias.
"""
